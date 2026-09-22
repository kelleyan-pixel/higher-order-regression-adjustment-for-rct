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
SWEEP = [(f"R2_{r:.2f}", "prev", ("large_hte", 2 * np.sqrt(r / (1 - r))), 8100 + i) for i, r in enumerate(R2_GRID)]

# battery: correction-study families ("prev", p = 5) and the scalar lognormal family of the introduction
# ("ex", p = 1: Y(0) = 0.9 X + (1 + c X) u, Y(1) = Y(0) + 1 + g X), weighted towards small R^2_tau
BATTERY = [
    # key, engine, parameters, seed, label, group
    ("gauss_low",   "prev", ("gauss_iid", 0.3),      8201, "Gaussian", "small"),
    ("misspec_null", "prev", ("misspec", 0.0),       8203, "Misspecified, $G=0$", "small"),
    ("skew_null",   "prev", ("null_hte", 0.0),       8204, "Skewed, het., $G=0$", "small"),
    ("heavy_low",   "prev", ("heavy_het", 0.5),      8208, "Heavy-tailed, het.", "small"),
    ("ln12_small",  "ex",   (1.2, 1.0, 0.3),         8301, "Lognormal $s=1.2$", "small"),
    ("ln15_small",  "ex",   (1.5, 1.0, 0.1),         8302, "Lognormal $s=1.5$", "small"),
    ("ln15_g05",    "ex",   (1.5, 1.0, 0.5),         8303, "Lognormal $s=1.5$", "small"),
    ("heavy_ex",    "ex",   (1.8, 1.5, 0.3),         8304, "Lognormal $s=1.8$, $\\sigma=1+1.5X$", "small"),
    ("ln20_small",  "ex",   (2.0, 1.0, 0.1),         8305, "Lognormal $s=2$", "small"),
    ("ln15_mid",    "ex",   (1.5, 1.0, 1.0),         8306, "Lognormal $s=1.5$", "moderate"),
    ("rare",        "prev", ("rare_extreme", 1.0),   8210, "Rare extreme values", "moderate"),
    ("heavy_high",  "prev", ("heavy_het", 3.0),      8209, "Heavy-tailed, het.", "large"),
    ("ln15_large",  "ex",   (1.5, 1.0, 3.0),         8307, "Lognormal $s=1.5$", "large"),
    ("gauss_high",  "prev", ("large_hte", 3.0),      8202, "Gaussian", "large"),
]
LEVELS = [0.80, 0.90, 0.95, 0.98, 0.99]


def summarize(a, tr, n):
    tau = tr["PATE"]
    Gn = tr["G"] / n
    se = {"IREG_HC1": a["I_se1"], "IREG_HC3": a["I_se3"],
          "IREG_HC1c": np.sqrt(a["I_se1"] ** 2 + a["plug"]), "IREG_HC1o": np.sqrt(a["I_se1"] ** 2 + Gn),
          "IREG_HC3c": np.sqrt(a["I_se3"] ** 2 + a["plug"]),
          "REG_HC1": a["R_se1"], "REG_HC3": a["R_se3"]}
    out = {"coverage": {}, "mcse": {}, "median_width": {}, "calibration": {f"{L:.2f}": {} for L in LEVELS}}
    for k, s in se.items():
        est, q = (a["R_est"], a["R_q"]) if k.startswith("REG") else (a["I_est"], a["I_q"])
        ok = np.isfinite(est) & np.isfinite(s)
        cov = np.where(ok, np.abs(est - tau) <= q * s, False)            # no finite interval -> non-cover
        out["coverage"][k] = float(cov.mean())
        out["mcse"][k] = float(np.sqrt(cov.mean() * (1 - cov.mean()) / len(cov)))
        out["median_width"][k] = float(np.median((2 * q * s)[ok]))
        df = a["R_df"] if k.startswith("REG") else a["I_df"]
        for L in LEVELS:
            qL = stats.t.ppf((1 + L) / 2, df)
            out["calibration"][f"{L:.2f}"][k] = float(np.where(ok, np.abs(est - tau) <= qL * s, False).mean())
    ok = np.isfinite(a["I_est"]) & np.isfinite(a["R_est"])
    eI, eR = (a["I_est"][ok] - tau) ** 2, (a["R_est"][ok] - tau) ** 2
    r = eI.mean() / eR.mean()
    out["mse_ratio"] = float(r)
    out["mse_ratio_mcse"] = float(np.std(eI - r * eR, ddof=1) / (eR.mean() * np.sqrt(ok.sum())))
    out["plug_over_Gn"] = float(np.nanmean(a["plug"]) / Gn) if Gn > 0 else None
    out["plug_mean"] = float(np.nanmean(a["plug"]))
    out["frac_rank_deficient"] = float(np.mean(a["rankdef"])) if "rankdef" in a else 0.0
    out["frac_leverage_one"] = float(np.mean(a["lev1"] & a.get("I_okc", np.ones_like(a["lev1"])))) if "lev1" in a else 0.0
    return out


def simulate(row):
    key, engine, par, seed = row[:4]
    if engine == "prev":
        fam, g = par
        cp.GSTAR[fam] = float(g)
        tr = cp.truth_prev(fam, P)
        a = cp.run_prev(fam, N, P, REPS, seed)
        p = P
    else:
        sx, c, g = par
        tr = cp.truth_example(sx, c, g)
        a, _ = cp.run_example(sx, c, g, N, REPS, seed)
        p = 1
    for lab, k in (("R", p + 2), ("I", 2 * p + 2)):                # regression columns: REG 1,X,W; IREG 1,X,W,WX
        a.setdefault(lab + "_df", np.full(len(a[lab + "_est"]), float(N - k)))
        a.setdefault(lab + "_q", stats.t.ppf(0.975, a[lab + "_df"]))
    return tr, a, p


def run(designs, fname):
    res = {"n": N, "reps": REPS, "engine": "NumPy (correction_python.py)", "levels": LEVELS, "designs": {}}
    for row in designs:
        tr, a, p = simulate(row)
        s = summarize(a, tr, N)
        s.update(engine=row[1], parameters=list(row[2]), p=p, seed=row[3], G=tr["G"], V_star=tr["V_star"], R2_tau=tr["R2_tau"],
                 gamma_star=float(row[2][-1]))
        if len(row) > 4:
            s["label"], s["group"] = row[4], row[5]
        res["designs"][row[0]] = s
        print(row[0], f"R2={tr['R2_tau']:.3f}", {k: round(v, 4) for k, v in s["coverage"].items()}, f"mse_ratio={s['mse_ratio']:.3f}",
              flush=True)
    with open(os.path.join(OUT, fname), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "sweep"):
        run(SWEEP, "hte_sweep.json")
    if which in ("all", "battery"):
        run(BATTERY, "hte_battery.json")
