"""Standard vs superpopulation-corrected IREG intervals: Python designs.

Two groups of designs, each reproduced on exactly the datasets that produced the
reported uncorrected coverages:

* ``prev``: supplementary applied designs from an earlier version of this package
  (Tables 1-4 of that release: p = 5, n = 500, 20,000 replications; and the
  dimension sweep, n = 1000, p in {2, 5, 10}, 10,000 replications).  The DGP, seed
  and fitting code below are copied verbatim from that release's
  simulations/run_inference.py; they include the moderate-HTE skewed designs and
  the large-HTE design.
* ``example``: the introductory example and its heavy-tailed variant of the current
  paper (p = 1, n = 500, 100,000 replications), copied from run_example.py.

Usage: python simulations/correction_python.py [prev|example]
Output: results/correction/py_<design>.json (design-level summaries)
"""
import json, os, sys
import numpy as np
from scipy import stats
from correction_common import summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "correction")

# ------------------------------------------------------------------ supplementary designs (verbatim)
LEV_TOL = 1e-10
RCOND_TOL = 1e-10

SEEDS = {"gauss_iid": 101, "gauss_indep_arms": 102, "skew_het": 103, "null_hte": 104,
         "heavy_het": 105, "rare_extreme": 106, "large_hte": 107, "misspec": 108}


# ---------------------------------------------------------------- DGPs
def ar1_chol(p, rho):
    S = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    return np.linalg.cholesky(S)


def make_X(rng, name, B, n, p):
    if name in ("gauss_iid", "gauss_indep_arms"):
        return rng.standard_normal((B, n, p))
    if name in ("large_hte", "misspec"):
        return rng.standard_normal((B, n, p)) @ ar1_chol(p, 0.5).T
    if name in ("skew_het", "null_hte", "heavy_het"):
        s = 1.5 if name == "heavy_het" else 1.1
        U = rng.standard_normal((B, n, p))
        U[..., 0] = np.exp(s * U[..., 0]) - np.exp(s ** 2 / 2)
        varU = np.ones(p)
        varU[0] = (np.exp(s ** 2) - 1) * np.exp(s ** 2)
        L = ar1_chol(p, 0.35)
        X = U @ L.T
        return X / np.sqrt((L ** 2) @ varU)          # population standardisation
    if name == "rare_extreme":
        q = 0.02
        a, b = np.sqrt(q / (1 - q)), np.sqrt((1 - q) / q)
        X = rng.standard_normal((B, n, p))
        X[..., 0] = np.where(rng.random((B, n)) < q, b, -a)
        return X
    raise ValueError(name)


GSTAR = dict(gauss_iid=0.3, gauss_indep_arms=0.3, skew_het=0.3, null_hte=0.0,
             heavy_het=0.5, rare_extreme=1.0, large_hte=3.0, misspec=0.0)


def dgp(rng, name, B, n, p, tau=1.0):
    """Returns X, W, Y, SATE."""
    X = make_X(rng, name, B, n, p)
    m = n // 2
    idx = np.argsort(rng.random((B, n)), 1)
    W = np.take_along_axis(np.r_[np.ones(m), np.zeros(n - m)][None].repeat(B, 0), idx, 1)
    gam = np.zeros(p)
    gam[0] = GSTAR[name]
    if name in ("skew_het", "null_hte"):
        sig = 1 + 0.5 * np.abs(X[..., 0]) + 0.25 * np.linalg.norm(X, axis=-1)
    elif name == "heavy_het":
        sig = 1 + 0.75 * np.abs(X[..., 0]) + 0.25 * np.linalg.norm(X, axis=-1)
    else:
        sig = np.ones(X.shape[:-1])
    base = X @ np.full(p, 0.9)
    if name == "misspec":
        base = base + 0.20 * (X[..., 0] ** 2 - 1) + 0.15 * X[..., 0] * X[..., 1]
    e1 = sig * rng.standard_normal((B, n))
    if name == "gauss_indep_arms":                   # eps(1), eps(0) independent
        e0 = sig * rng.standard_normal((B, n))
    else:                                            # shared noise: eps(1) = eps(0)
        e0 = e1
    Y1 = base + X @ gam + tau + e1
    Y0 = base + e0
    Y = np.where(W == 1, Y1, Y0)
    return X, W, Y, (Y1 - Y0).mean(1)


# ---------------------------------------------------------------- estimation
def fit(D, Y, col):
    """Batched OLS with HC1/HC2/HC3 standard errors of coefficient `col`."""
    B, n, k = D.shape
    XtX = np.einsum("bij,bik->bjk", D, D)
    ok = (1.0 / np.linalg.cond(XtX)) > RCOND_TOL
    with np.errstate(all="ignore"):
        A = np.linalg.pinv(XtX)
        coef = np.einsum("bjk,bik,bi->bj", A, D, Y)
        res = Y - np.einsum("bij,bj->bi", D, coef)
        h = np.einsum("bij,bjk,bik->bi", D, A, D)
        c = np.einsum("bk,bik->bi", A[:, col, :], D)        # tau_hat = c'Y
        omh = 1.0 - h
        lev = omh <= LEV_TOL                                 # estimatr: zero contribution
        den = np.where(lev, 1.0, omh)
        w2 = np.where(lev, 0.0, res ** 2 / den)
        w3 = np.where(lev, 0.0, res ** 2 / den ** 2)
        se = {"HC1": np.sqrt(np.einsum("bi,bi->b", c ** 2, res ** 2) * n / (n - k)),
              "HC2": np.sqrt(np.einsum("bi,bi->b", c ** 2, w2)),
              "HC3": np.sqrt(np.einsum("bi,bi->b", c ** 2, w3))}
    return coef, se, lev.any(1), ok




def run_prev(name, n, p, reps, seed, chunk=250, tau=1.0):
    """Same loop, chunking and draw order as the original simulate() for these designs."""
    rng = np.random.default_rng(seed)
    keys = ["R_est", "R_se1", "R_se3", "I_est", "I_se1", "I_se2", "I_se3", "plug", "lev1", "rankdef",
            "I_okc"]
    acc = {k: [] for k in keys}
    done = 0
    while done < reps:
        B = min(chunk, reps - done)
        X, W, Y, sate = dgp(rng, name, B, n, p, tau)
        one = np.ones((B, n, 1))
        cR, sR, levR, okR = fit(np.concatenate([one, X, W[..., None]], -1), Y, p + 1)
        Xc = X - X.mean(1, keepdims=True)
        DI = np.concatenate([one, Xc, W[..., None], W[..., None] * Xc], -1)
        cI, sI, levI, okI = fit(DI, Y, p + 1)
        Sx = np.einsum("bij,bik->bjk", Xc, Xc) / (n - 1)
        delta = cI[:, p + 2:]
        acc["plug"].append(np.einsum("bj,bjk,bk->b", delta, Sx, delta) / n)
        nan = lambda v, ok: np.where(ok, v, np.nan)          # singular fit -> no interval
        acc["R_est"].append(nan(cR[:, p + 1], okR)); acc["I_est"].append(nan(cI[:, p + 1], okI))
        acc["R_se1"].append(sR["HC1"]); acc["R_se3"].append(sR["HC3"])
        for k in ("1", "2", "3"):
            acc["I_se" + k].append(sI["HC" + k])
        acc["lev1"].append(levI); acc["rankdef"].append(~okI); acc["I_okc"].append(okI)
        done += B
    a = {k: np.concatenate(v) for k, v in acc.items()}
    a["R_df"] = np.full(reps, float(n - p - 2)); a["I_df"] = np.full(reps, float(n - 2 * p - 2))
    a["R_q"] = stats.t.ppf(0.975, a["R_df"]); a["I_q"] = stats.t.ppf(0.975, a["I_df"])
    return a


def truth_prev(name, p):
    """gamma, Sigma_X, G, V* = G + 4 E[eps^2], PATE for the correction-study designs."""
    gam = np.zeros(p); gam[0] = GSTAR[name]
    if name in ("large_hte", "misspec"):
        S = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    elif name in ("skew_het", "null_hte", "heavy_het"):
        s = 1.5 if name == "heavy_het" else 1.1
        varU = np.ones(p); varU[0] = (np.exp(s ** 2) - 1) * np.exp(s ** 2)
        L = ar1_chol(p, 0.35); C = L @ np.diag(varU) @ L.T; d = np.sqrt(np.diag(C))
        S = C / np.outer(d, d)
    else:
        S = np.eye(p)
    G = float(gam @ S @ gam)
    # E[eps_i^2]: oracle residual variance, by a large Monte Carlo draw where needed
    rng = np.random.default_rng(99)
    if name in ("skew_het", "null_hte", "heavy_het"):
        X = make_X(rng, name, 1, 1_000_000, p)[0]
        c1, c2 = (0.75, 0.25) if name == "heavy_het" else (0.5, 0.25)
        Eeps2 = float(np.mean((1 + c1 * np.abs(X[:, 0]) + c2 * np.linalg.norm(X, axis=1)) ** 2))
        how = "Monte Carlo (10^6 draws)"
    elif name == "misspec":
        X = make_X(rng, name, 1, 1_000_000, p)[0]
        g = 0.20 * (X[:, 0] ** 2 - 1) + 0.15 * X[:, 0] * X[:, 1]
        Eeps2 = float(np.var(g) + 1.0); how = "Monte Carlo (10^6 draws)"
    else:
        Eeps2, how = 1.0, "exact"
    V = G + 4 * Eeps2
    return dict(gamma=gam.tolist(), Sigma_X=S.tolist(), G=G, V_star=V, R2_tau=G / V, PATE=1.0,
                V_star_method=how)


# ------------------------------------------------------------------ current-paper examples
def run_example(sx, c, g, n, reps, seed, chunk=1000):
    """Same draws as run_example.coverage(); adds HC2c/HC3c and the plug-in itself."""
    rng = np.random.default_rng(seed)
    m, tau = n // 2, 1.0 + g * np.exp(sx ** 2 / 2)
    acc = {k: [] for k in ["R_est", "R_se1", "R_se3", "I_est", "I_se1", "I_se2", "I_se3", "plug"]}
    for _ in range(reps // chunk):
        B = chunk
        X = np.exp(sx * rng.standard_normal((B, n)))
        idx = np.argsort(rng.random((B, n)), 1)
        W = np.take_along_axis(np.r_[np.ones(m), np.zeros(m)][None].repeat(B, 0), idx, 1)
        Y = 0.9 * X + W * (1.0 + g * X) + (1 + c * X) * rng.standard_normal((B, n))
        one = np.ones((B, n)); Xc = X - X.mean(1, keepdims=True)
        for lab, D in [("R", np.stack([one, X, W], -1)), ("I", np.stack([one, Xc, W, W * Xc], -1))]:
            A = np.linalg.inv(np.einsum("bij,bik->bjk", D, D))
            cvec = np.einsum("bk,bik->bi", A[:, 2, :], D)
            coef = np.einsum("bjk,bik,bi->bj", A, D, Y)
            res = Y - np.einsum("bij,bj->bi", D, coef)
            h = np.einsum("bij,bjk,bik->bi", D, A, D)
            acc[lab + "_est"].append(np.einsum("bi,bi->b", cvec, Y))
            acc[lab + "_se1"].append(np.sqrt(np.einsum("bi,bi->b", cvec ** 2, res ** 2) * n / (n - D.shape[-1])))
            acc[lab + "_se3"].append(np.sqrt(np.einsum("bi,bi->b", cvec ** 2, res ** 2 / (1 - h) ** 2)))
            if lab == "I":
                acc["I_se2"].append(np.sqrt(np.einsum("bi,bi->b", cvec ** 2, res ** 2 / (1 - h))))
                acc["plug"].append(coef[:, 3] ** 2 * np.var(X, 1, ddof=1) / n)
    a = {k: np.concatenate(v) for k, v in acc.items()}
    a["R_df"] = np.full(reps, float(n - 3)); a["I_df"] = np.full(reps, float(n - 4))
    a["R_q"] = stats.t.ppf(0.975, a["R_df"]); a["I_q"] = stats.t.ppf(0.975, a["I_df"])
    a["lev1"] = np.zeros(reps, bool); a["rankdef"] = np.zeros(reps, bool)
    return a, tau


def truth_example(sx, c, g):
    M = lambda k: np.exp(k ** 2 * sx ** 2 / 2)
    var = M(2) - M(1) ** 2
    G = g ** 2 * var
    Es2 = 1 + 2 * c * M(1) + c ** 2 * M(2)
    V = G + 4 * Es2
    return dict(gamma=[g], Sigma_X=[[var]], G=G, V_star=V, R2_tau=G / V, PATE=1.0 + g * M(1),
                V_star_method="exact")


def main(which):
    os.makedirs(OUT, exist_ok=True)
    if which == "prev":
        plan = [(nm, 500, 5, 20000, SEEDS[nm], "prev_table1") for nm in
                ["gauss_iid", "gauss_indep_arms", "skew_het", "null_hte", "heavy_het", "rare_extreme", "large_hte"]]
        plan += [(nm, 1000, p, 10000, SEEDS[nm] * 1000 + p, "prev_table3") for nm in
                 ["gauss_iid", "skew_het", "misspec", "large_hte"] for p in (2, 5, 10)]
        for nm, n, p, reps, seed, src in plan:
            key = f"{nm}_n{n}_p{p}"
            f = os.path.join(OUT, f"py_{key}.json")
            if os.path.exists(f):
                continue
            a = run_prev(nm, n, p, reps, seed)
            t = truth_prev(nm, p)
            meta = dict(design=key, source=src, engine="python (supplementary-design code)", n=n, p=p,
                        seed=seed, hte=t["G"] > 0, truth=t)
            s = summarize(a, 1.0, t["G"], n, meta)
            # conditional-on-estimable IREG HC1 coverage, as originally reported for these designs
            ok = a["I_okc"]
            q = a["I_q"][0]
            s["check_prev_release"] = {"IREG_HC1_conditional": float(np.mean(np.abs(a["I_est"][ok] - 1) <= q * a["I_se1"][ok]))}
            json.dump(s, open(f, "w"), indent=1)
            print(key, s["coverage"]["IREG_HC1"]["coverage"], s["coverage"]["IREG_HC1c"]["coverage"], flush=True)
    elif which == "example":
        for key, (sx, c, g, seed) in {"intro_example": (1.1, 1.0, 0.0, 17),
                                      "heavy_tail_example": (1.8, 1.5, 0.3, 31)}.items():
            f = os.path.join(OUT, f"py_{key}.json")
            if os.path.exists(f):
                continue
            a, tau = run_example(sx, c, g, 500, 100000, seed)
            t = truth_example(sx, c, g)
            meta = dict(design=key, source="current paper (run_example.py)", engine="python", n=500, p=1,
                        seed=seed, hte=True, truth=t)
            s = summarize(a, tau, t["G"], 500, meta)
            json.dump(s, open(f, "w"), indent=1)
            print(key, s["coverage"]["IREG_HC1"]["coverage"], s["coverage"]["IREG_HC1c"]["coverage"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prev")
