"""Section 3.4 and 3.5 simulations: a controlled treatment-effect-heterogeneity sweep and a battery of
designs in which finite-sample leverage effects and the superpopulation component G/n can coexist.

Both reuse the correction-study engine of correction_python.py (covariates, noise, balanced complete
randomization, OLS with HC1/HC3, the plug-in term) and change only the treatment-effect slope gamma*
of each design family. Truth (G, V*, R^2_tau, PATE) comes from correction_python.truth_prev.

  python3 simulations/hte_designs.py        # writes results/hte_sweep.json and results/hte_battery.json
"""
import json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correction_python as cp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
N, P, REPS = 500, 5, 20000

# controlled sweep: Gaussian AR(0.5) covariates, homoskedastic N(0,1) noise, well specified;
# G = gamma*^2 and E[eps^2] = 1, so R^2_tau = gamma*^2 / (gamma*^2 + 4)
R2_GRID = [0.0, 0.1, 0.25, 0.5, 0.7, 0.85, 0.9]
SWEEP = [(f"R2_{r:.2f}", "large_hte", 2 * np.sqrt(r / (1 - r)), 8100 + i) for i, r in enumerate(R2_GRID)]

# battery: existing design families, some with a larger slope than in the correction study
BATTERY = [
    ("gauss_low",        "gauss_iid",    0.3, 8201, "Gaussian, small HTE"),
    ("gauss_high",       "large_hte",    3.0, 8202, "Gaussian, large HTE"),
    ("misspec_null",     "misspec",      0.0, 8203, "Gaussian, misspecified, $G=0$"),
    ("skew_null",        "null_hte",     0.0, 8204, "Skewed, heteroskedastic, $G=0$"),
    ("skew_low",         "skew_het",     0.3, 8205, "Skewed, heteroskedastic, small HTE"),
    ("skew_mid",         "skew_het",     1.5, 8206, "Skewed, heteroskedastic, moderate HTE"),
    ("skew_high",        "skew_het",     3.0, 8207, "Skewed, heteroskedastic, large HTE"),
    ("heavy_low",        "heavy_het",    0.5, 8208, "Heavy-tailed, heteroskedastic, small HTE"),
    ("heavy_high",       "heavy_het",    3.0, 8209, "Heavy-tailed, heteroskedastic, large HTE"),
    ("rare",             "rare_extreme", 1.0, 8210, "Rare extreme values, moderate HTE"),
]


def summarize(a, tr, n):
    tau = tr["PATE"]
    Gn = tr["G"] / n
    se = {"IREG_HC1": a["I_se1"], "IREG_HC3": a["I_se3"],
          "IREG_HC1c": np.sqrt(a["I_se1"] ** 2 + a["plug"]), "IREG_HC1o": np.sqrt(a["I_se1"] ** 2 + Gn),
          "IREG_HC3c": np.sqrt(a["I_se3"] ** 2 + a["plug"]),
          "REG_HC1": a["R_se1"], "REG_HC3": a["R_se3"]}
    out = {"coverage": {}, "mcse": {}, "median_width": {}}
    for k, s in se.items():
        est, q = (a["R_est"], a["R_q"]) if k.startswith("REG") else (a["I_est"], a["I_q"])
        ok = np.isfinite(est) & np.isfinite(s)
        cov = np.where(ok, np.abs(est - tau) <= q * s, False)            # no finite interval -> non-cover
        out["coverage"][k] = float(cov.mean())
        out["mcse"][k] = float(np.sqrt(cov.mean() * (1 - cov.mean()) / len(cov)))
        out["median_width"][k] = float(np.median((2 * q * s)[ok]))
    ok = np.isfinite(a["I_est"]) & np.isfinite(a["R_est"])
    eI, eR = (a["I_est"][ok] - tau) ** 2, (a["R_est"][ok] - tau) ** 2
    r = eI.mean() / eR.mean()
    out["mse_ratio"] = float(r)
    out["mse_ratio_mcse"] = float(np.std(eI - r * eR, ddof=1) / (eR.mean() * np.sqrt(ok.sum())))
    out["plug_over_Gn"] = float(np.nanmean(a["plug"]) / Gn) if Gn > 0 else None
    out["plug_mean"] = float(np.nanmean(a["plug"]))
    out["frac_rank_deficient"] = float(np.mean(a["rankdef"]))
    out["frac_leverage_one"] = float(np.mean(a["lev1"] & a["I_okc"]))
    return out


def run(designs, fname):
    res = {"n": N, "p": P, "reps": REPS, "engine": "NumPy (correction_python.py)", "designs": {}}
    for row in designs:
        key, fam, g, seed = row[:4]
        cp.GSTAR[fam] = float(g)
        tr = cp.truth_prev(fam, P)
        a = cp.run_prev(fam, N, P, REPS, seed)
        s = summarize(a, tr, N)
        s.update(family=fam, gamma_star=float(g), seed=seed, G=tr["G"], V_star=tr["V_star"], R2_tau=tr["R2_tau"])
        if len(row) > 4:
            s["label"] = row[4]
        res["designs"][key] = s
        print(key, f"R2={tr['R2_tau']:.3f}", {k: round(v, 4) for k, v in s["coverage"].items()}, f"mse_ratio={s['mse_ratio']:.3f}",
              flush=True)
    with open(os.path.join(OUT, fname), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    run(SWEEP, "hte_sweep.json")
    run(BATTERY, "hte_battery.json")
