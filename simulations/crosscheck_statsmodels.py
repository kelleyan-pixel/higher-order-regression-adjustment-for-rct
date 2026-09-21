"""Cross-check of the NumPy OLS/HC implementation (hc.py) against statsmodels.

hc.py is the independent Python implementation used to cross-check the R/estimatr
simulation engine (crosscheck_r.py).  Here it is compared with statsmodels on one
simulated skewed, heteroskedastic dataset, for REG and IREG and for HC1, HC2, HC3.

Usage: python simulations/crosscheck_statsmodels.py
"""
import json, os
import numpy as np
import statsmodels.api as sm
from hc import fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")


def main(seed=20260921, n=200, p=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, 0] = np.exp(1.1 * X[:, 0])
    W = rng.permutation(np.r_[np.ones(n // 2), np.zeros(n - n // 2)])
    Y = X @ np.full(p, 0.9) + 0.3 * W * X[:, 0] + W + (1 + 0.5 * X[:, 0]) * rng.standard_normal(n)
    Xc = X - X.mean(0)
    specs = {"REG": np.column_stack([np.ones(n), W, X]),
             "IREG": np.column_stack([np.ones(n), W, Xc, W[:, None] * Xc])}
    rows = {}
    for lab, D in specs.items():
        coef, se, lev, ok = fit(D[None], Y[None], 1)
        ours = dict(coef=float(coef[0, 1]), **{k: float(v[0]) for k, v in se.items()})
        theirs = {"coef": float(sm.OLS(Y, D).fit().params[1])}
        for hc in ("HC1", "HC2", "HC3"):
            theirs[hc] = float(np.sqrt(sm.OLS(Y, D).fit(cov_type=hc).cov_params()[1, 1]))
        rel = {k: abs(ours[k] - theirs[k]) / max(abs(theirs[k]), 1e-12) for k in ours}
        rows[lab] = dict(ours=ours, statsmodels=theirs, max_rel_diff=max(rel.values()))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "crosscheck.json"), "w"), indent=1)
    worst = max(r["max_rel_diff"] for r in rows.values())
    print(f"max relative difference across all quantities: {worst:.2e}")
    assert worst < 1e-10, "cross-check failed"


if __name__ == "__main__":
    main()
