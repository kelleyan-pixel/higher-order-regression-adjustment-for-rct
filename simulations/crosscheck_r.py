"""Cross-check of the R/estimatr engine against two independent Python implementations.

r/validate_estimatr.R exports simulated datasets (results/r_datasets/*.csv) together
with estimatr's output for REG = lm_robust(y ~ w + x) and IREG = lm_lin(y ~ w, ~ x)
under HC1, HC2 and HC3.  This script refits every dataset with (a) the NumPy
implementation in hc.py and (b) statsmodels, and compares point estimates, standard
errors, degrees of freedom, confidence intervals and IREG leverages.

Datasets containing a leverage-one observation are compared on HC1 and leverages
only, because HC2/HC3 at leverage one are 0/0 and implementations handle that case
differently (documented in VERIFICATION.md).

Usage: python simulations/crosscheck_r.py
"""
import glob, json, os
import numpy as np
import statsmodels.api as sm
from scipy import stats
from hc import fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "results", "r_datasets")


def main(tol=1e-8):
    files = sorted(glob.glob(os.path.join(DIR, "*.csv")))
    assert files, "run r/validate_estimatr.R first"
    worst = {"est": 0.0, "se": 0.0, "df": 0.0, "ci": 0.0, "leverage": 0.0, "statsmodels_se": 0.0}
    n_lev1 = 0
    for f in files:
        data = np.genfromtxt(f, delimiter=",", names=True)
        R = json.load(open(f[:-4] + ".json"))
        y, w = data["y"], data["w"]
        X = np.column_stack([data[k] for k in data.dtype.names if k.startswith("x")])
        n, p = X.shape
        Xc = X - X.mean(0)
        designs = {"REG": np.column_stack([np.ones(n), w, X]),
                   "IREG": np.column_stack([np.ones(n), w, Xc, w[:, None] * Xc])}
        D = designs["IREG"]
        h = np.einsum("ij,jk,ik->i", D, np.linalg.inv(D.T @ D), D)
        worst["leverage"] = max(worst["leverage"], np.max(np.abs(h - np.array(R["leverage_IREG"]))))
        lev1 = h.max() > 1 - 1e-8
        n_lev1 += lev1
        for lab, M in designs.items():
            coef, se, _, ok = fit(M[None], y[None], 1)
            k = M.shape[1]
            for hc in (("HC1",) if lev1 else ("HC1", "HC2", "HC3")):
                r = R[hc][lab]
                q = stats.t.ppf(0.975, n - k)
                ci = (coef[0, 1] - q * se[hc][0], coef[0, 1] + q * se[hc][0])
                rel = lambda a, b: abs(a - b) / max(abs(b), 1.0)
                worst["est"] = max(worst["est"], rel(coef[0, 1], r["est"]))
                worst["se"] = max(worst["se"], rel(se[hc][0], r["se"]))
                worst["df"] = max(worst["df"], abs((n - k) - r["df"]))
                worst["ci"] = max(worst["ci"], rel(ci[0], r["ci"][0]), rel(ci[1], r["ci"][1]))
                sm_se = np.sqrt(sm.OLS(y, M).fit(cov_type=hc).cov_params()[1, 1])
                worst["statsmodels_se"] = max(worst["statsmodels_se"], rel(sm_se, r["se"]))
    out = {"datasets": len(files), "datasets_with_leverage_one": int(n_lev1),
           "max_rel_gap": worst, "tolerance": tol, "passed": all(v < tol for v in worst.values())}
    json.dump(out, open(os.path.join(ROOT, "results", "r_crosscheck.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    assert out["passed"], "R/Python cross-check failed"


if __name__ == "__main__":
    main()
