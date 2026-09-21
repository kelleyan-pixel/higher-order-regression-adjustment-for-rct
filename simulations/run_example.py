"""Introductory example and range-of-validity sweep (paper Section 1 and Table 9).

Scalar DGP:
    X  ~ LogNormal(0, sx^2)            (raw, uncentred: the estimators are
                                        invariant to translations of X)
    sigma(X) = 1 + c X
    Y(0) = 0.9 X + sigma(X) eta,   eta ~ N(0,1)
    Y(1) = Y(0) + 1 + g X
so that the PATE is tau = 1 + g E[X], the model is well specified
(E[eps | X] = 0) and the noise is heteroskedastic in X.

All population quantities entering D_p are available in closed form, so the
predicted MSE ratio 1 + Delta/n involves no Monte-Carlo error.

Usage: python run_example.py
"""
import json, os
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# ------------------------------------------------------------------ theory
def population(sx, c, g, qlev=None):
    """Population quantities for X ~ LogNormal(0, sx^2), sigma(X) = 1 + c X, slope g.  With qlev, X is
    truncated at its qlev quantile (moments E[X^k | X <= x_q] in closed form); qlev=None is the full law."""
    if qlev is None:
        M = lambda k: np.exp(k ** 2 * sx ** 2 / 2)      # raw lognormal moments
    else:
        from scipy.stats import norm
        lc = sx * norm.ppf(qlev)
        M = lambda k: np.exp(k ** 2 * sx ** 2 / 2) * norm.cdf((lc - k * sx ** 2) / sx) / norm.cdf(lc / sx)
    mu, var = M(1), M(2) - M(1) ** 2
    sd = np.sqrt(var)
    c3 = M(3) - 3 * mu * M(2) + 2 * mu ** 3
    c4 = M(4) - 4 * mu * M(3) + 6 * mu ** 2 * M(2) - 3 * mu ** 4
    s, ku = c3 / sd ** 3, c4 / sd ** 4                  # skewness, kurtosis of Z
    gam = g * sd
    G = gam ** 2
    Es2 = 1 + 2 * c * mu + c ** 2 * M(2)                # E[sigma^2]
    EZs2 = 2 * c * sd + c ** 2 * (M(3) - mu * M(2)) / sd  # E[Z sigma^2]
    shape = G * (ku - 4 * s ** 2 - 5)                   # -(2p+3)G + F - Q - 3R, p=1
    het = 8 * s * EZs2                                  # 8 S' E[Z eps^2]
    V = G + 4 * Es2
    return dict(skew=s, kurt=ku, sd=sd, mu=mu, G=G, V=V, R2tau=G / V,
                shape=shape, het=het, Dp=shape + het, Delta=(shape + het) / V)


# ------------------------------------------------------------------ simulation
def mse_ratio(sx, c, g, n, reps, seed=0, chunk=2000):
    rng = np.random.default_rng(seed)
    m, tau = n // 2, 1.0 + g * np.exp(sx ** 2 / 2)
    num = den = 0.0
    batches = []
    for _ in range(reps // chunk):
        X1 = np.exp(sx * rng.standard_normal((chunk, m)))
        X0 = np.exp(sx * rng.standard_normal((chunk, m)))
        Y1 = 0.9 * X1 + 1.0 + g * X1 + (1 + c * X1) * rng.standard_normal((chunk, m))
        Y0 = 0.9 * X0 + (1 + c * X0) * rng.standard_normal((chunk, m))

        def st(X, Y):
            xb, yb = X.mean(1), Y.mean(1)
            Xc, Yc = X - xb[:, None], Y - yb[:, None]
            return xb, yb, (Xc * Xc).sum(1), (Xc * Yc).sum(1)

        x1, y1, S1, s1 = st(X1, Y1)
        x0, y0, S0, s0 = st(X0, Y0)
        D = x1 - x0
        eR = y1 - y0 - D * (s1 + s0) / (S1 + S0) - tau            # REG
        eI = y1 - y0 - D * (s1 / S1 + s0 / S0) / 2 - tau          # IREG
        batches.append([np.mean(eI ** 2), np.mean(eR ** 2)])
        num += np.sum(eI ** 2)
        den += np.sum(eR ** 2)
    b = np.array(batches)
    r = b[:, 0] / b[:, 1]
    return float(num / den), float(r.std(ddof=1) / np.sqrt(len(r)))


def coverage(sx, c, g, n, reps, seed=0, chunk=1000):
    """HC1/HC2/HC3 and corrected-HC1 coverage of the PATE for REG and IREG."""
    rng = np.random.default_rng(seed)
    m, tau = n // 2, 1.0 + g * np.exp(sx ** 2 / 2)
    acc = {k: [] for k in ["eR", "eI", "R1", "R3", "I1", "I2", "I3", "Ic"]}
    for _ in range(reps // chunk):
        B = chunk
        X = np.exp(sx * rng.standard_normal((B, n)))
        idx = np.argsort(rng.random((B, n)), 1)
        W = np.take_along_axis(np.r_[np.ones(m), np.zeros(m)][None].repeat(B, 0), idx, 1)
        Y = 0.9 * X + W * (1.0 + g * X) + (1 + c * X) * rng.standard_normal((B, n))
        one = np.ones((B, n))
        Xc = X - X.mean(1, keepdims=True)
        out = {}
        for lab, D in [("R", np.stack([one, X, W], -1)),
                       ("I", np.stack([one, Xc, W, W * Xc], -1))]:
            k = D.shape[-1]
            A = np.linalg.inv(np.einsum("bij,bik->bjk", D, D))
            cvec = np.einsum("bk,bik->bi", A[:, 2, :], D)
            coef = np.einsum("bjk,bik,bi->bj", A, D, Y)
            res = Y - np.einsum("bij,bj->bi", D, coef)
            h = np.einsum("bij,bjk,bik->bi", D, A, D)
            out[lab] = (np.einsum("bi,bi->b", cvec, Y),
                        np.sqrt(np.einsum("bi,bi->b", cvec ** 2, res ** 2) * n / (n - k)),
                        np.sqrt(np.einsum("bi,bi->b", cvec ** 2, res ** 2 / (1 - h))),
                        np.sqrt(np.einsum("bi,bi->b", cvec ** 2, res ** 2 / (1 - h) ** 2)),
                        coef)
        acc["eR"].append(out["R"][0] - tau); acc["eI"].append(out["I"][0] - tau)
        acc["R1"].append(out["R"][1]); acc["R3"].append(out["R"][3])
        acc["I1"].append(out["I"][1]); acc["I2"].append(out["I"][2]); acc["I3"].append(out["I"][3])
        acc["Ic"].append(np.sqrt(out["I"][1] ** 2 + out["I"][4][:, 3] ** 2 * np.var(X, 1, ddof=1) / n))
    a = {k: np.concatenate(v) for k, v in acc.items()}
    # t critical values with n-k degrees of freedom, as in estimatr: k=3 for REG, k=4 for IREG
    qR, qI = stats.t.ppf(0.975, n - 3), stats.t.ppf(0.975, n - 4)
    res = {}
    for lab, e, se in [("REG+HC1", "eR", "R1"), ("REG+HC3", "eR", "R3"),
                       ("IREG+HC1", "eI", "I1"), ("IREG+HC2", "eI", "I2"),
                       ("IREG+HC3", "eI", "I3"), ("IREG+HC1c", "eI", "Ic")]:
        q = qR if e == "eR" else qI
        cov = np.abs(a[e]) <= q * a[se]
        res[lab] = (float(cov.mean()), float(np.sqrt(cov.var() / cov.size)))
    res["rmse"] = (float(np.sqrt(np.mean(a["eR"] ** 2))), float(np.sqrt(np.mean(a["eI"] ** 2))))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}

    # ---- introductory example: sx = 0.8, c = 2, g = 0.4
    sx, c, g = 0.8, 2.0, 0.4
    pop = population(sx, c, g)
    res["intro_population"] = pop
    res["intro_ratio"] = {}
    for n, reps in [(500, 200000), (2000, 100000)]:
        r, se = mse_ratio(sx, c, g, n, reps, seed=n)
        res["intro_ratio"][n] = dict(sim=r, se=se, predicted=1 + pop["Delta"] / n)
        print("ratio", n, res["intro_ratio"][n], flush=True)
    res["intro_coverage"] = coverage(sx, c, g, 500, 100000, seed=17)
    print("coverage", res["intro_coverage"], flush=True)

    # ---- range of validity: increasing skewness at fixed c, g
    res["sweep"] = {}
    for sxx in [0.6, 0.8, 1.0, 1.2, 1.5]:
        p = population(sxx, 1.0, 0.4)
        row = dict(skew=p["skew"], kurt=p["kurt"], Delta=p["Delta"], R2tau=p["R2tau"])
        for n, reps in [(500, 100000), (2000, 50000), (8000, 20000)]:
            r, se = mse_ratio(sxx, 1.0, 0.4, n, reps, seed=1000 + n)
            row[n] = dict(sim=r, se=se, predicted=1 + p["Delta"] / n)
        res["sweep"][sxx] = row
        print("sweep", sxx, row, flush=True)

    # ---- the heavy-tailed cautionary case (sx = 1.8, c = 1.5, g = 0.3)
    p = population(1.8, 1.5, 0.3)
    row = dict(population=p)
    for n, reps in [(500, 100000), (2000, 50000), (8000, 20000)]:
        r, se = mse_ratio(1.8, 1.5, 0.3, n, reps, seed=2000 + n)
        row[n] = dict(sim=r, se=se, predicted=1 + p["Delta"] / n)
    row["coverage"] = coverage(1.8, 1.5, 0.3, 500, 100000, seed=31)
    res["heavy_tail_case"] = row
    print("heavy", row, flush=True)

    json.dump(res, open(os.path.join(OUT, "example.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
