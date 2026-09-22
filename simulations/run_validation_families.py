"""Further verification of Theorem 2: four covariate families of the initial draft at p = 2, 5, 8.

Families (whitened coordinates; the estimators are invariant to invertible linear maps of the
covariates, so the AR(1) correlation of the draft enters only through gamma and, for 'misspec',
through tr(Sigma^2)):
  gauss     Z ~ N(0, I_p), well specified, homoskedastic         D_p = -(p+1) G
  lowkurt   Z iid Rademacher (kurtosis 1)                          D_p = -(p+3) G
  heavytail Z = s xi / sqrt(E s^2), xi ~ N(0, I_p), s^2 in {1, K^2} w.p. {1-eps, eps}
                                                                   D_p = G [(p+2) kt - (2p+3)],  kt = E s^4 / (E s^2)^2
  misspec   Z ~ N(0, I_p), Y(w) adds c (Z' Sigma Z - tr Sigma), Sigma = AR(1) with rho
                                                                   D_p = -(p+1) G + 32 c^2 tr(Sigma^2)
All families have a shared, homoskedastic N(0,1) noise and gamma = sqrt(G) (1,...,1)/sqrt(p).
The control-variate mean n^2 E[d1 L] = gamma' K gamma - G is exact (K = E[Z Z' ||Z||^2]).

  python3 simulations/run_validation_families.py          # writes results/validation_families.json
"""
import json, os, sys
import numpy as np
from sim import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
EPS, KSC, C_MIS, RHO = 0.02, 6.0, 0.3, 0.3
ES2, ES4 = (1 - EPS) + EPS * KSC ** 2, (1 - EPS) + EPS * KSC ** 4
KT = ES4 / ES2 ** 2
G = {"gauss": 0.49, "lowkurt": 1.96, "heavytail": 0.49, "misspec": 0.49}


def ar1(p):
    i = np.arange(p)
    return RHO ** np.abs(i[:, None] - i[None, :])


def design(fam, p):
    g = np.full(p, np.sqrt(G[fam] / p))
    zero = lambda Z: np.zeros(Z.shape[:-1]); one = lambda Z: np.ones(Z.shape[:-1])
    d = dict(p=p, gamma=g, g1=zero, g0=zero, s1=one, s0=one, kinds=["gauss"] * p)
    Gv = G[fam]
    if fam == "gauss":
        target, K = -(p + 1) * Gv, (p + 2)
    elif fam == "lowkurt":
        d["kinds"] = ["rade"] * p
        target, K = -(p + 3) * Gv, p
    elif fam == "heavytail":
        def draw(rng, shape):
            B, m, pp = shape
            s = np.where(rng.random((B, m, 1)) < EPS, KSC, 1.0)
            return s * rng.standard_normal(shape) / np.sqrt(ES2)
        d["draw"] = draw
        target, K = Gv * ((p + 2) * KT - (2 * p + 3)), (p + 2) * KT
    elif fam == "misspec":
        S = ar1(p); trS = np.trace(S)
        q = lambda Z: C_MIS * (np.einsum("...i,ij,...j->...", Z, S, Z) - trS)
        d["g1"] = d["g0"] = q
        target, K = -(p + 1) * Gv + 32 * C_MIS ** 2 * np.trace(S @ S), (p + 2)
    d["target"], d["cv_mean"] = float(target), float(K * Gv - Gv)       # K is a multiple of I_p in all four
    return d


def matrix_formula_check(fam, p, N=10 ** 6, seed=7):
    """Theorem 2's matrix form evaluated with Monte Carlo moments (a check on the closed forms)."""
    rng = np.random.default_rng(seed); d = design(fam, p); g = d["gamma"]
    Z = d["draw"](rng, (1, N, p))[0] if "draw" in d else (rng.integers(0, 2, (N, p)) * 2.0 - 1 if fam == "lowkurt" else rng.standard_normal((N, p)))
    eps = d["g1"](Z) + rng.standard_normal(N)
    nz = (Z ** 2).sum(1); zg = Z @ g
    K = np.einsum("ni,nj,n->ij", Z, Z, nz) / N; S = (Z * nz[:, None]).mean(0)
    Sg = np.einsum("ni,nj,n->ij", Z, Z, zg) / N; E = np.einsum("ni,nj,n->ij", Z, Z, eps) / N
    Ze2 = (Z * (eps ** 2)[:, None]).mean(0)                    # E_W = 0: shared residual
    Dp = -(2 * p + 3) * g @ g + g @ K @ g - (g @ S) ** 2 - 3 * (Sg ** 2).sum() + 8 * np.trace(E @ E) + 8 * S @ Ze2
    return float(Dp)


PLAN_LEVEL = [(f, p) for f in ("gauss", "lowkurt", "heavytail", "misspec") for p in (2, 5, 8)]
N_LEVEL, B_LEVEL = 4000, 20000
PLAN_RATE_N, B_RATE = (500, 1000, 2000), 20000                         # p = 5; n = 4000 comes from the level run

if __name__ == "__main__":
    res = {"n_level": N_LEVEL, "B_level": B_LEVEL, "B_rate": B_RATE, "level": {}, "rate": {}}
    for i, (fam, p) in enumerate(PLAN_LEVEL):
        d = design(fam, p)
        raw, cv = run(d, N_LEVEL, B_LEVEL, 500, seed=9100 + i)
        est = cv.mean() + 2 * d["cv_mean"]; se = cv.std(ddof=1) / np.sqrt(len(cv))
        res["level"][f"{fam}_p{p}"] = dict(family=fam, p=p, G=G[fam], target=d["target"], mc_matrix_check=matrix_formula_check(fam, p),
                                           est=float(est), se=float(se), z=float((est - d["target"]) / se),
                                           raw=float(raw.mean()), se_raw=float(raw.std(ddof=1) / np.sqrt(len(raw))))
        print(f"{fam}_p{p}", {k: round(v, 3) for k, v in res["level"][f"{fam}_p{p}"].items() if isinstance(v, float)}, flush=True)
    for fam in ("gauss", "lowkurt", "heavytail", "misspec"):
        d = design(fam, 5)
        for j, n in enumerate(PLAN_RATE_N):
            raw, cv = run(d, n, B_RATE, max(500, 800000 // n), seed=9200 + 10 * j + ("gauss", "lowkurt", "heavytail", "misspec").index(fam))
            est = cv.mean() + 2 * d["cv_mean"]; se = cv.std(ddof=1) / np.sqrt(len(cv))
            res["rate"][f"{fam}_n{n}"] = dict(family=fam, n=n, target=d["target"], est=float(est), se=float(se), z=float((est - d["target"]) / se))
            print(f"rate {fam} n={n}: est {est:.2f} se {se:.2f} z {(est - d['target']) / se:+.2f}", flush=True)
    with open(os.path.join(OUT, "validation_families.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
