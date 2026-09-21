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
        (f"{ip['skew']:.2f}", "skewness $3.69$"),
        (f"{ip['kurt']:.1f}", "kurtosis $34.4$"),
        (f"{ip['R2tau']:.4f}", "is $0.0032$"),
        (f"{ip['Delta']:.1f}", "$\\Delta = 15.6$"),
        (f"{100 * (ex['intro_ratio']['500']['predicted'] - 1):.1f}", "by $3.1\\%$"),
        (f"{100 * (ex['intro_ratio']['500']['sim'] - 1):.1f}", "simulation gives $2.3\\%$"),
        (f"{100 * ex['intro_ratio']['500']['se']:.2f}", "error $0.02\\%$"),
        (f"{100 * (ex['intro_ratio']['2000']['predicted'] - 1):.2f}", "prediction is $0.78\\%$"),
        (f"{100 * (ex['intro_ratio']['2000']['sim'] - 1):.2f}", "simulation gives $0.67\\%$"),
        (f"{100 * ex['intro_coverage']['IREG+HC1'][0]:.1f}", "ATE in $94.5\\%$ of replications"),
        (f"{100 * ex['intro_coverage']['REG+HC1'][0]:.1f}", "REG+HC1 in $95.0\\%$"),
        (f"{hr['population']['skew']:.0f}", "skewness $136$"),
        (f"{hr['population']['Delta'] / 1e4:.1f}", "3.9\\times10^4"),
        (f"{hr['500']['predicted']:.1f}", "ratio of $78.5$"),
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
        (all(cal(g0, "IREG_HC1c", l) < 100 * float(l) for l in ("0.80", "0.90"))
         and all(cal(g0, "IREG_HC1c", l) > 100 * float(l) for l in ("0.98", "0.99")),
         "That agreement with nominal is a crossing point, not calibration"),
        (abs(W(big)["median_width_ratio_uncorrected_misses"] - W(big)["median_width_ratio_uncorrected_covers"]) < 0.02,
         "the lengthening is nearly uniform"),
        (all(W(d)["median_width_ratio_uncorrected_misses"] > 1.3 and W(d)["median_width_ratio_uncorrected_covers"] < 1.1
             for d in (hte, g0)), "the lengthening is concentrated where HC1 fails"),
        (max(sweep[k]["Delta"] for k in ("0.6", "0.8", "1.0")) <= 50 and ip["Delta"] <= 50,
         "In our moderately skewed designs the deficiency amounts to at most a few dozen observations"),
        (len(favors_reg) >= 2, "in several of our skewed, heteroskedastic designs it favors REG"),
        (C[big]["truth"]["R2_tau"] > 0.5 and (C[big]["p"] + 1) * C[big]["truth"]["R2_tau"] < 10,
         "a real but modest second-order gain"),
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
         "the simulated comparison has the sign the second-order analysis predicts"),
        # Section 3.3 connection and Appendix C tail sensitivity
        (all(abs(1 + population_(s_, 1.0, 0.0)["Delta"] / 500 - C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] / C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"]) < 0.015
             for s_ in (0.6, 0.8)), "they agree closely for $s\\le0.8$"),
        (all(C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] > C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"]
             and population_(s_, 1.0, 0.0)["Delta"] > 0 for s_ in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)), "the simulated ratio exceeds one in every design, as predicted"),
        (all(10 <= population_(sx, c_, g)["Delta"] / population_(sx, c_, g, 1 - 1e-4)["Delta"] <= 1000
             for sx, c_, g in ((1.8, 1.5, 0.3), (1.5, 1.0, 0.4), (2.0, 1.0, 0.0))), "by one to three orders of magnitude"),
        (all(0.5 <= (1 + population_(sx, c_, g, 1 - 1e-4)["Delta"] / 500) / sim <= 2 for sx, c_, g, sim in
             ((1.8, 1.5, 0.3, hr["500"]["sim"]), (1.5, 1.0, 0.4, ex["sweep"]["1.5"]["500"]["sim"]),
              (2.0, 1.0, 0.0, C[g0]["sampling"]["IREG"]["mse"] / C[g0]["sampling"]["REG"]["mse"]))), "to the same order of magnitude as the simulated ratio"),
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
