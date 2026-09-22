"""Check the numbers quoted in the prose against the stored results.

Numbers in Section 3 and Appendix D are generated (paper/tables/gamma0_values.tex).
The remaining prose numbers (introduction, Appendices B and C) are typed in the
source; this script recomputes each of them from results/*.json and asserts that
(i) the recomputed, rounded value equals the value in the text and (ii) that text
actually occurs in paper/main.tex.

Usage: python simulations/check_prose_numbers.py
"""
import json, os, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_example import population as population_
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
TEX = open(os.path.join(ROOT, "paper", "main.tex")).read()


def main():
    ex, va, ab, crt, bi = R("example.json"), R("validation.json"), R("ablation.json"), \
        R("crt_correction.json"), R("bias.json")
    ip, hr = ex["intro_population"], ex["heavy_tail_case"]
    B = math.sqrt(400000)
    g3 = va["gauss3_homo"]["est"]
    full = va["cexp2_full"]
    z3200 = {k: (v["est"]["3200"]["cv"] - v["formula"]) / v["est"]["3200"]["se_cv"] for k, v in va.items()}
    z200 = {k: (v["est"]["200"]["cv"] - v["formula"]) / v["est"]["200"]["se_cv"] for k, v in va.items()}
    b800 = [abs(v[f"z_{e}"]) for k, v in bi.items() if k.endswith("n800") for e in ("REG", "IREG")]
    bf = bi["cexp2_full_n200"]
    s08 = ex["sweep"]["0.8"]
    checks = [
        # (recomputed string, exact text that must appear in main.tex)
        (f"{hr['500']['sim']:.2f}", "ratio is $1.68$"),
        (f"{hr['2000']['sim']:.2f}", "$1.34$ at $n=2{,}000$"),
        (f"{hr['8000']['sim']:.2f}", "$1.19$ at $n=8{,}000$"),
        (f"{hr['500']['predicted']:.1f}", "predictions of $78.5$"),
        (f"{hr['2000']['predicted']:.1f}", "$20.4$ and"),
        (f"{hr['8000']['predicted']:.1f}", "and $5.8$"),
        (f"{hr['population']['R2tau']:.3f}", "R^2_\\tau\\approx0.009"),
        (f"{100 * hr['coverage']['IREG+HC1'][0]:.1f}", "$\\tau$ in $85.6\\%$"),
        (f"{100 * hr['coverage']['IREG+HC2'][0]:.1f}", "IREG+HC2 in $88.9\\%$"),
        (f"{100 * hr['coverage']['IREG+HC3'][0]:.1f}", "IREG+HC3 in $94.0\\%$"),
        (f"{100 * hr['coverage']['REG+HC1'][0]:.1f}", "against $94.4\\%$ for REG+HC1"),
        (f"{100 * hr['coverage']['REG+HC3'][0]:.1f}", "$96.7\\%$ for REG+HC3"),
        (f"{hr['500']['predicted'] / hr['500']['sim']:.0f}", "more than forty"),
        (f"{ex['sweep']['1.5']['500']['predicted'] / ex['sweep']['1.5']['500']['sim']:.0f}", "factor of four"),
        (f"{g3['200']['se_raw'] * B:.0f}", "from $151$"),
        (f"{g3['800']['se_raw'] * B:.0f}", "to $293$"),
        (f"{g3['200']['se_cv'] * B:.0f}", "is $56$"),
        (f"{g3['800']['se_cv'] * B:.0f}", "and $55$"),
        (f"{g3['3200']['se_raw'] / g3['3200']['se_cv']:.0f}", "ten times smaller"),
        (f"{max(abs(z) for z in z200.values()):.0f}", "up to $22$"),
        (f"{full['formula'] - full['est']['200']['cv']:.1f}", "from $12.1$"),
        (f"{full['formula'] - full['est']['800']['cv']:.1f}", "to $5.7$"),
        (f"{full['formula'] - full['est']['3200']['cv']:.1f}", "to $2.6$"),
        (f"{max(abs(z) for z in z3200.values()):.1f}", "within $2.6$ standard errors"),
        (f"{z3200['cexp2_full']:+.1f}", "are $-2.6$"),
        (f"{z3200['cexp2_het_null']:+.1f}", "and $+2.3$"),
        (f"{4 * full['G']:.1f}", "D_p - 3.2"),
        (f"{ab['gauss3_homo'][[k for k in ab['gauss3_homo'] if 'Bern' in k][0]]:.0f}", "by up to thirty standard errors"),
        (f"{crt['gauss3_homo_n200']['diff']:.1f}\\pm{crt['gauss3_homo_n200']['se']:.1f}", "$4.3\\pm0.6$"),
        (f"{crt['gauss3_homo_n800']['diff']:.1f}\\pm{crt['gauss3_homo_n800']['se']:.1f}", "$6.3\\pm1.6$"),
        (f"{crt['cexp2_homo_n200']['diff']:.1f}\\pm{crt['cexp2_homo_n200']['se']:.1f}", "$3.0\\pm1.1$"),
        (f"{crt['cexp2_homo_n800']['diff']:.1f}\\pm{crt['cexp2_homo_n800']['se']:.1f}", "$8.6\\pm3.0$"),
        (f"{max(b800):.1f}", "within $0.8$ Monte Carlo"),
        (f"{100 * (bf['bias_REG'] - bf['pred_REG']) / abs(bf['pred_REG']):.0f}", "about $11\\%$"),
        (f"{bf['z_REG']:.1f}", "($z=9.3$)"),
        (f"{bi['cexp2_full_n800']['z_REG']:.1f}", "($z=0.4$)"),
    ]
    expected = {  # the value each text snippet asserts, as it should be recomputed
        "more than forty": lambda v: float(v) > 40,
        "factor of four": lambda v: round(float(v)) == 4,
        "ten times smaller": lambda v: round(float(v)) == 10,
        "by up to thirty standard errors": lambda v: 28 <= float(v) <= 32,
        "3.9\\times10^4": lambda v: v == "3.9",
    }
    _res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    SW = json.load(open(os.path.join(_res, "hte_sweep.json")))["designs"]
    BA = json.load(open(os.path.join(_res, "hte_battery.json")))["designs"]
    rho_, m_ = 0.02, 250
    q0_, q1_ = (1 - rho_) ** m_, m_ * rho_ * (1 - rho_) ** (m_ - 1)
    lev_th, rd_th = q1_ * (2 * (1 - q0_) - q1_), 1 - (1 - q0_) ** 2
    # statements in Appendix E about the calibration diagnostic (results/correction_summary.json)
    C = R("correction_summary.json")["designs"]
    lev = ["0.80", "0.90", "0.95", "0.98", "0.99"]
    cal = lambda d, m, l: 100 * C[d]["calibration"][l][m]["coverage"]
    hte, g0 = "heavy_tail_example", "lognormal_s2.0_prop"
    shift = [cal(hte, "IREG_HC1c", l) - cal(hte, "IREG_HC1", l) for l in lev]
    oshift = [cal(hte, "IREG_HC1o", l) - cal(hte, "IREG_HC1", l) for l in lev]
    hc3err = [abs(cal(d, "IREG_HC3", l) - 100 * float(l)) for d in (hte, g0) for l in lev]
    hc3c_over = all(cal(d, "IREG_HC3c", l) > 100 * float(l) for d in (hte, g0) for l in lev)
    g0_oracle_eq = all(cal(g0, "IREG_HC1o", l) == cal(g0, "IREG_HC1", l) for l in lev)
    for ok, text in [(5 <= min(shift) and max(shift) <= 10, "by five to ten points depending on the level"),
                     (max(abs(x) for x in oshift) <= 1.2, "the oracle moves HC1 coverage by about a point"),
                     (max(hc3err) <= 1.2, "HC3 by itself is within about a point of nominal at every level"),
                     (hc3c_over, "the plug-in overcovers at every level"),
                     (g0_oracle_eq, "the oracle changes nothing")]:
        checks.append(("ok" if ok else "FAILED", text))
        expected[text] = lambda v: v == "ok"
    # synthesis and width statements (Section 3.2, Appendix E, Discussion)
    big = "large_hte_n500_p5"
    W = lambda d: C[d]["width"]["HC1"]
    sweep = ex["sweep"]
    favors_reg = [k for k in ("cexp2_het_null", "cexp2_full", "mixed2_het") if va[k]["formula"] > 0]
    for ok, text in [
        (all(abs(cal(big, m, l) - 100 * float(l)) <= 0.5 for m in ("IREG_HC1c", "IREG_HC1o") for l in lev),
         "the feasible and oracle intervals are close to nominal at every level"),
        (all(cal(d, "IREG_HC1c", "0.80") < 80 and cal(d, "IREG_HC1c", "0.99") > 99 for d in (hte, g0)),
         "is a crossing point rather than evidence of a calibrated procedure"),
        (abs(W(big)["median_width_ratio_uncorrected_misses"] - W(big)["median_width_ratio_uncorrected_covers"]) < 0.02,
         "the lengthening is nearly uniform"),
        (all(W(d)["median_width_ratio_uncorrected_misses"] > 1.3 and W(d)["median_width_ratio_uncorrected_covers"] < 1.1
             for d in (hte, g0)), "the lengthening is concentrated where HC1 fails"),
        (all(v["Delta"] > 0 for v in sweep.values()) and sweep["1.2"]["Delta"] > 100,
         "ranges from a few observations to more than a hundred in our scalar designs with skewed covariates"),
        (abs(sweep["1.0"]["8000"]["predicted"] - sweep["1.0"]["8000"]["sim"]) < abs(sweep["1.0"]["500"]["predicted"] - sweep["1.0"]["500"]["sim"]),
         "but the agreement improves with $n$"),
        # Section 3.3-3.4 diagnostics
        (all(abs(C[f"lognormal_s{s_}_prop"]["median_width_95"]["IREG_HC1"] / C[f"lognormal_s{s_}_prop"]["median_width_95"]["REG_HC1"] - 1) < 0.03
             for s_ in ("0.6", "0.8", "1.0", "1.2", "1.5", "1.8", "2.0")), "The median HC1 widths of IREG and REG are nearly equal throughout"),
        (C[g0]["studentized"]["IREG_HC1"]["implied_scale_from_95"] > 1.2
         and max(abs(v["observed"] - v["scale_predicted"]) for v in C[g0]["studentized"]["IREG_HC1"]["levels"].values()) < 0.02,
         "HC1's failure in this design is primarily an understatement of the variance scale"),
        (abs(C[g0]["studentized"]["IREG_HC3"]["implied_scale_from_95"] - 1) < 0.1
         and C[g0]["studentized"]["IREG_HC3"]["levels"]["0.99"]["quantile_ratio"] > C[g0]["studentized"]["IREG_HC3"]["levels"]["0.80"]["quantile_ratio"],
         "HC3 largely removes the scale error"),
        (all(C[f"lognormal_s{s_}_prop"]["se_ratio_HC3_HC1"]["REG"]["median"] < C[f"lognormal_s{s_}_prop"]["se_ratio_HC3_HC1"]["IREG"]["median"]
             for s_ in ("1.0", "1.2", "1.5", "1.8", "2.0")), "for REG the corresponding values at"),
        (R("r_gamma0_rare.json")["rare_q0.02_homo"]["IREG_HC1"]["leverage_one"]["coverage"] > 0.94, "with $\\sigma=1$, IREG+HC1 covers"),
        (R("r_transform_check.json")["log1pX"]["IREG_HC1"]["median_width"] > R("r_transform_check.json")["X"]["IREG_HC1"]["median_width"],
         "but the median interval widens from"),
        # variability versus standard error, and the MSE comparison (Sections 3.3 and 4, Appendix C)
        (C[g0]["sampling"]["IREG"]["sd"] / C[g0]["sampling"]["REG"]["sd"] > 1.2
         and abs(C[g0]["sampling"]["IREG"]["median_se_HC1"] / C[g0]["sampling"]["REG"]["median_se_HC1"] - 1) < 0.05,
         "in this design IREG is genuinely more variable, and HC1 does not register the difference"),
        (C[g0]["sampling"]["REG"]["sd"] > C[g0]["sampling"]["REG"]["median_se_HC1"]
         and abs(R("r_gamma0_skew.json")["lognormal_s2.0_prop"]["REG_HC1"]["all"]["coverage"] - 0.95) < 0.006,
         "the Monte Carlo standard deviations exceed the median standard errors even for REG"),
        (all(0 < d1 < d2 for d1, d2 in zip(*[[1 + population_(s_, 1.0, 0.0)["Delta"] / 500
              - C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] / C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"]
              for s_ in L_] for L_ in ((1.0, 1.2, 1.5, 1.8), (1.2, 1.5, 1.8, 2.0))])),
         "the prediction exceeds the simulated ratio by a growing margin"),
        (all(v[n]["sim"] > 1 and v[n]["predicted"] > 1 for v in sweep.values() for n in ("500", "2000", "8000"))
         and hr["500"]["sim"] > 1 and hr["500"]["predicted"] > 1,
         "The sign of the comparison agrees with the expansion in every design where we compare the two"),
        (all((v["formula"] > 0) == (v["est"]["3200"]["cv"] > 0) for v in va.values())
         and all((population_(s_, 1.0, 0.0)["Delta"] > 0) == (C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] > C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"])
                 for s_ in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)),
         "and the heavy-tailed example), so the decomposition"),
        # leading-variance comparison, Table 7 convergence, Table 6 exact probabilities
        (abs(C["lognormal_s0.6_prop"]["sampling"]["REG"]["sd"] / (population_(0.6, 1.0, 0.0)["V"] / 500) ** 0.5 - 1) < 0.02
         and C["lognormal_s2.0_prop"]["sampling"]["REG"]["sd"] < 0.6 * (population_(2.0, 1.0, 0.0)["V"] / 500) ** 0.5,
         "matches $\\sqrt{V^*/n}$ at $s=0.6$"),
        (abs(full["est"]["3200"]["cv"] - (full["formula"] - 4 * full["G"])) < full["est"]["3200"]["se_cv"],
         "lies within one standard error of the estimate at $n=3{,}200$"),
        (abs((lambda e2, e8, e32: e32 + (e32 - e8) * ((e32 - e8) / (e8 - e2)) / (1 - (e32 - e8) / (e8 - e2)))(
             full["est"]["200"]["cv"], full["est"]["800"]["cv"], full["est"]["3200"]["cv"]) - full["formula"]) < 1.0,
         "which extrapolates to about"),
        # introduction (typed numbers)
        (f"{100 * (ex['intro_ratio']['500']['sim'] - 1):.1f}", "by $6.7\\%$ at $n=500$"),
        (f"{ex['intro_population']['Delta']:.0f}", "around $83$ users"),
        (f"{100 * ex['intro_coverage']['IREG+HC1'][0]:.1f}", "undercovers at $93.5\\%$"),
        (f"{ex['intro_population']['skew']:.1f}", "skewness $8.2$"),
        (ex["intro_coverage"]["IREG+HC3"][0] > 0.945 and all(abs(ex["intro_coverage"][k][0] - 0.95) < 0.01 for k in ("REG+HC1", "REG+HC3")),
         "(fixed by switching to HC3); meanwhile, simple regression is well-calibrated"),
        # Section 3.3
        (all(R("r_gamma0_skew.json")[f"lognormal_s{s_}_prop"]["IREG_HC3"]["all"]["coverage"] - R("r_gamma0_skew.json")[f"lognormal_s{s_}_prop"]["IREG_HC1"]["all"]["coverage"] > 0.03 for s_ in ("1.5", "1.8", "2.0"))
         and all(R("r_gamma0_skew.json")[f"lognormal_s{s_}_prop"]["IREG_HC3"]["all"]["coverage"] < 0.945 for s_ in ("1.8", "2.0")),
         "HC3 substantially improves IREG's coverage"),
        (C[g0]["sampling"]["mse_ratio"] > 1.5, "yet IREG's MSE exceeds REG's by a factor of"),
        # Section 3.4
        (all(abs(v - 0.95) < 0.01 for v in SW["R2_0.00"]["coverage"].values()), "With $R^2_\\tau=0$ every interval is close to nominal"),
        (all(abs(d["coverage"]["IREG_HC1"] - (2 * norm.cdf(1.96 * math.sqrt(1 - d["R2_tau"])) - 1)) < 0.01 for d in SW.values()),
         "tracking its asymptotic coverage"),
        (0 <= SW["R2_0.90"]["coverage"]["IREG_HC3"] - SW["R2_0.90"]["coverage"]["IREG_HC1"] < 0.01, "barely more than HC1"),
        (all(d["coverage"]["IREG_HC1"] < d["coverage"]["IREG_HC1o"] - 0.005 for k, d in SW.items() if k != "R2_0.00")
         and all(abs(d["coverage"]["IREG_HC1o"] - 0.95) < 0.01 for d in SW.values()), "The oracle correction restores nominal coverage at every level"),
        # Section 3.5
        (all(abs(BA[k][m]["coverage"] if False else BA[k]["coverage"][m] - 0.95) < 0.01 for k in ("gauss_low", "gauss_high") for m in ("IREG_HC1c", "REG_HC1")),
         "In the Gaussian designs both procedures are close to nominal"),
        (BA["skew_null"]["coverage"]["IREG_HC1c"] > BA["skew_null"]["coverage"]["IREG_HC1"], "the correction raises IREG's coverage from"),
        (BA["skew_low"]["coverage"]["IREG_HC1"] > BA["skew_mid"]["coverage"]["IREG_HC1"] > BA["skew_high"]["coverage"]["IREG_HC1"]
         and all(abs(BA[k]["coverage"]["IREG_HC1c"] - BA[k]["coverage"]["IREG_HC1o"]) < 0.005 for k in ("skew_low", "skew_mid", "skew_high")),
         "the feasible correction restores"),
        (min(BA[k]["coverage"]["REG_HC1"] for k in ("skew_low", "skew_mid")) > 0.945 and BA["skew_high"]["coverage"]["REG_HC1"] < 0.94,
         "REG+HC1 stays near nominal at small and moderate heterogeneity"),
        (BA["heavy_high"]["coverage"]["REG_HC1"] < 0.93 and BA["heavy_high"]["coverage"]["REG_HC1"] < BA["heavy_high"]["coverage"]["IREG_HC1c"],
         "The heavy-tailed designs show the mechanisms compounding"),
        (BA["rare"]["coverage"]["IREG_HC1c"] < 0.945 and BA["rare"]["coverage"]["IREG_HC1c"] > BA["rare"]["coverage"]["IREG_HC1"], "still below nominal"),
        (min(BA[k]["mse_ratio"] for k in ("skew_null", "skew_low", "heavy_low")) > 1.02 and all(abs(BA[k]["mse_ratio"] - 1) < 0.01 for k in ("gauss_low", "gauss_high", "skew_high"))
         and BA["heavy_high"]["mse_ratio"] > 1.05 and BA["rare"]["mse_ratio"] < 0.98,
         "IREG remains less precise in the heavy-tailed design even with large heterogeneity"),
        (C[g0]["calibration"]["0.80"]["REG_HC1"]["coverage"] < 0.80 and C[g0]["calibration"]["0.99"]["REG_HC1"]["coverage"] > 0.99
         and C[g0]["studentized"]["REG_HC1"]["levels"]["0.80"]["quantile_ratio"] > 1 > C[g0]["studentized"]["REG_HC1"]["levels"]["0.99"]["quantile_ratio"]
         and abs(C[g0]["calibration"]["0.80"]["IREG_HC3"]["coverage"] - 0.80) < abs(C[g0]["calibration"]["0.80"]["REG_HC1"]["coverage"] - 0.80),
         "REG+HC1 shows a milder form of the same crossing pattern"),
        (all(abs(v["frac_leverage_one_IREG"] - lev_th) < 3 * (lev_th * (1 - lev_th) / v["reps"]) ** 0.5
             and abs(v["frac_rank_deficient_IREG"] - rd_th) < 3 * (rd_th * (1 - rd_th) / v["reps"]) ** 0.5
             for v in R("r_gamma0_rare.json").values()), "the exact probabilities are"),
        # Section 3.3 connection and Appendix C tail sensitivity
        (all(abs(1 + population_(s_, 1.0, 0.0)["Delta"] / 500 - C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] / C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"]) < 0.015
             for s_ in (0.6, 0.8)), "they agree closely for $s\\le0.8$"),
        (all(C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] > C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"]
             and population_(s_, 1.0, 0.0)["Delta"] > 0 for s_ in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)), "the simulated ratio exceeds one in every design, as predicted"),
        (all(10 <= population_(sx, c_, g)["Delta"] / population_(sx, c_, g, 1 - 1e-4)["Delta"] <= 1000
             for sx, c_, g in ((1.8, 1.5, 0.3), (1.5, 1.0, 0.4), (2.0, 1.0, 0.0))), "by one to three orders of magnitude"),
        (all(1 + population_(sx, c_, g, 1 - 1e-8)["Delta"] / 500 >= 3 * sim for sx, c_, g, sim in
             ((1.8, 1.5, 0.3, hr["500"]["sim"]), (1.5, 1.0, 0.4, ex["sweep"]["1.5"]["500"]["sim"]),
              (2.0, 1.0, 0.0, C[g0]["sampling"]["IREG"]["mse"] / C[g0]["sampling"]["REG"]["mse"]))),
         "gives predictions several times the simulated ratio"),
    ]:
        checks.append(("ok" if ok else "FAILED", text))
        expected[text] = lambda v: v == "ok"
    bad = []
    for val, text in checks:
        if text not in TEX:
            bad.append(f"text not found in main.tex: {text!r}")
            continue
        ok = expected[text](val) if text in expected else (val.lstrip("+") in text.replace("{,}", ","))
        if not ok:
            bad.append(f"{text!r}: recomputed {val}")
    print(f"checked {len(checks)} prose numbers; {len(bad)} problems")
    for b in bad:
        print("  ", b)
    assert not bad


if __name__ == "__main__":
    main()
