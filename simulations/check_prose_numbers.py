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
        (f"{hr['500']['sim']:.2f}", "simulated MSE ratios of $1.68$"),
        (f"{hr['2000']['sim']:.2f}", "$1.68$, $1.34$, and"),
        (f"{hr['8000']['sim']:.2f}", "$1.34$, and $1.19$"),
        (f"{hr['500']['predicted']:.1f}", "prediction is then $78.5$"),
        (f"{hr['2000']['predicted']:.1f}", "$78.5$, $20.4$, and"),
        (f"{hr['8000']['predicted']:.1f}", "and $5.8$"),
        (f"{hr['population']['R2tau']:.3f}", "R_\\tau^2\\approx0.009"),
        (f"{100 * hr['coverage']['IREG+HC1'][0]:.1f}", "IREG is $85.6\\%$ with HC1"),
        (f"{100 * hr['coverage']['IREG+HC2'][0]:.1f}", "$88.9\\%$ with HC2"),
        (f"{100 * hr['coverage']['IREG+HC3'][0]:.1f}", "$94.0\\%$ with HC3"),
        (f"{100 * hr['coverage']['REG+HC1'][0]:.1f}", "$94.4\\%$ for REG+HC1"),
        (f"{100 * hr['coverage']['REG+HC3'][0]:.1f}", "$96.7\\%$ for REG+HC3"),
        (f"{ex['sweep']['1.5']['500']['predicted'] / ex['sweep']['1.5']['500']['sim']:.0f}", "the predicted ratio is four times the simulated ratio"),
        (f"{g3['200']['se_raw'] * B:.0f}", "from $151$"),
        (f"{g3['800']['se_raw'] * B:.0f}", "to $293$"),
        (f"{g3['200']['se_cv'] * B:.0f}", "is $56$"),
        (f"{g3['800']['se_cv'] * B:.0f}", "and $55$"),
        (f"{g3['3200']['se_raw'] / g3['3200']['se_cv']:.0f}", "ten times smaller"),
        (f"{full['formula'] - full['est']['800']['cv']:.1f}", "to $5.7$"),
        (f"{max(abs(z) for z in z3200.values()):.1f}", "within $2.6$ standard errors"),
        (f"{4 * full['G']:.1f}", "D_\\Bern=D_p-3.2"),
        (f"{ab['gauss3_homo'][[k for k in ab['gauss3_homo'] if 'Bern' in k][0]]:.0f}", "as many as thirty standard errors"),
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
        "the predicted ratio is four times the simulated ratio": lambda v: round(float(v)) == 4,
        "ten times smaller": lambda v: round(float(v)) == 10,
        "as many as thirty standard errors": lambda v: 28 <= float(v) <= 32,
        "3.9\\times10^4": lambda v: v == "3.9",
    }
    _res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    SW = json.load(open(os.path.join(_res, "hte_sweep.json")))["designs"]
    FAM = json.load(open(os.path.join(_res, "validation_families.json")))
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
    for ok, text in [(5 <= min(shift) and max(shift) <= 10, "five to ten points depending on the nominal level"),
                     (max(abs(x) for x in oshift) <= 1.2, "changes HC1 coverage by only about one percentage point"),
                     (max(hc3err) <= 1.2, "HC3 alone is within about one percentage point of nominal"),
                     (hc3c_over, "then produces overcoverage"),
                     (g0_oracle_eq, "the oracle has no effect")]:
        checks.append(("ok" if ok else "FAILED", text))
        expected[text] = lambda v: v == "ok"
    # synthesis and width statements (Section 3.2, Appendix E, Discussion)
    big = "large_hte_n500_p5"
    W = lambda d: C[d]["width"]["HC1"]
    sweep = ex["sweep"]
    favors_reg = [k for k in ("cexp2_het_null", "cexp2_full", "mixed2_het") if va[k]["formula"] > 0]
    for ok, text in [
        # Discussion: in the most extreme designs both procedures undercover, and HC3 helps both
        (all(BA[k]["coverage"][m] < 0.95 for k in ("ln15_large", "heavy_high") for m in ("IREG_HC1c", "REG_HC1"))
         and all(BA[k]["coverage"]["IREG_HC3c"] > BA[k]["coverage"]["IREG_HC1c"]
                 and BA[k]["coverage"]["REG_HC3"] > BA[k]["coverage"]["REG_HC1"] for k in ("ln15_large", "heavy_high")),
         "HC3 improves coverage for both"),
        # Appendix B: without the control variate the Bernoulli constant is separated in no design
        (all(abs(v["est"]["3200"]["raw"] - (v["formula"] - 4 * v["G"])) <= 3 * v["est"]["3200"]["se_raw"] for v in va.values()),
         "($|z|\\leq3$ throughout)"),
        (__import__("run_validation_families").KSC == 6.0, "have six times the standard deviation"),
        (all(v["est"]["3200"]["se_raw"] / v["est"]["3200"]["se_cv"] >= 3.8 for v in va.values() if v["G"] > 0),
         "four to ten times larger in the heterogeneous designs"),
        (all(abs(cal(big, m, l) - 100 * float(l)) <= 0.5 for m in ("IREG_HC1c", "IREG_HC1o") for l in lev),
         "intervals have similar coverage across the nominal levels"),
        (all(cal(d, "IREG_HC1c", "0.80") < 80 and cal(d, "IREG_HC1c", "0.99") > 99 for d in (hte, g0)),
         "is therefore specific to that nominal level"),
        (abs(W(big)["median_width_ratio_uncorrected_misses"] - W(big)["median_width_ratio_uncorrected_covers"]) < 0.02,
         "the corresponding ratios are"),
        (all(W(d)["median_width_ratio_uncorrected_misses"] > 1.3 and W(d)["median_width_ratio_uncorrected_covers"] < 1.1
             for d in (hte, g0)), "among replications where the HC1 interval misses"),
        (all(v["Delta"] > 0 for v in sweep.values()) and sweep["1.2"]["Delta"] > 100,
         "ranges from a few observations to more than one hundred"),
        # Section 3.3-3.4 diagnostics
        (all(abs(C[f"lognormal_s{s_}_prop"]["median_width_95"]["IREG_HC1"] / C[f"lognormal_s{s_}_prop"]["median_width_95"]["REG_HC1"] - 1) < 0.03
             for s_ in ("0.6", "0.8", "1.0", "1.2", "1.5", "1.8", "2.0")), "The median HC1 widths of IREG and REG are nearly equal throughout"),
        (abs(C[g0]["studentized"]["IREG_HC3"]["implied_scale_from_95"] - 1) < 0.1
         and C[g0]["studentized"]["IREG_HC3"]["levels"]["0.99"]["quantile_ratio"] > C[g0]["studentized"]["IREG_HC3"]["levels"]["0.80"]["quantile_ratio"],
         "HC3 largely removes the variance-scale error"),
        (all(C[f"lognormal_s{s_}_prop"]["se_ratio_HC3_HC1"]["REG"]["median"] < C[f"lognormal_s{s_}_prop"]["se_ratio_HC3_HC1"]["IREG"]["median"]
             for s_ in ("1.0", "1.2", "1.5", "1.8", "2.0")), "For REG at $s=2$, the corresponding values are"),
        (R("r_gamma0_rare.json")["rare_q0.02_homo"]["IREG_HC1"]["leverage_one"]["coverage"] > 0.94, "$\\sigma=1$, coverage is"),
        (R("r_transform_check.json")["log1pX"]["IREG_HC1"]["median_width"] > R("r_transform_check.json")["X"]["IREG_HC1"]["median_width"],
         "The median interval width increases from"),
        # variability versus standard error, and the MSE comparison (Sections 3.3 and 4, Appendix C)
        (C[g0]["sampling"]["IREG"]["sd"] / C[g0]["sampling"]["REG"]["sd"] > 1.2
         and abs(C[g0]["sampling"]["IREG"]["median_se_HC1"] / C[g0]["sampling"]["REG"]["median_se_HC1"] - 1) < 0.05,
         "is not reflected in its HC1 standard errors"),
        (C[g0]["sampling"]["REG"]["sd"] > C[g0]["sampling"]["REG"]["median_se_HC1"]
         and abs(R("r_gamma0_skew.json")["lognormal_s2.0_prop"]["REG_HC1"]["all"]["coverage"] - 0.95) < 0.006,
         "also exceed their median reported standard errors"),
        (all(0 < d1 < d2 for d1, d2 in zip(*[[1 + population_(s_, 1.0, 0.0)["Delta"] / 500
              - C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] / C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"]
              for s_ in L_] for L_ in ((1.0, 1.2, 1.5, 1.8), (1.2, 1.5, 1.8, 2.0))])),
         "the prediction increasingly"),
        (all(v[n]["sim"] > 1 and v[n]["predicted"] > 1 for v in sweep.values() for n in ("500", "2000", "8000"))
         and hr["500"]["sim"] > 1 and hr["500"]["predicted"] > 1,
         "agrees with the second-order calculation in every"),
        (all((v["formula"] > 0) == (v["est"]["3200"]["cv"] > 0) for v in va.values())
         and all((population_(s_, 1.0, 0.0)["Delta"] > 0) == (C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] > C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"])
                 for s_ in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)),
         "The sign of the REG--IREG comparison"),
        # leading-variance comparison, Table 5 convergence, Table 15 exact probabilities
        (abs(C["lognormal_s0.6_prop"]["sampling"]["REG"]["sd"] / (population_(0.6, 1.0, 0.0)["V"] / 500) ** 0.5 - 1) < 0.02
         and C["lognormal_s2.0_prop"]["sampling"]["REG"]["sd"] < 0.6 * (population_(2.0, 1.0, 0.0)["V"] / 500) ** 0.5,
         "is close to the first-order benchmark $\\sqrt{V^*/n}$"),
        (abs(full["est"]["3200"]["cv"] - (full["formula"] - 4 * full["G"])) < full["est"]["3200"]["se_cv"],
         "standard error of the $n=3200$ estimate"),
        (abs((lambda e2, e8, e32: e32 + (e32 - e8) * ((e32 - e8) / (e8 - e2)) / (1 - (e32 - e8) / (e8 - e2)))(
             full["est"]["200"]["cv"], full["est"]["800"]["cv"], full["est"]["3200"]["cv"]) - full["formula"]) < 1.0,
         "giving an extrapolated value of about"),
        # Appendix B: further covariate families
        (max(FAM["level"].values(), key=lambda d: abs(d["mc_matrix_check"] - d["target"]) / abs(d["target"]))["family"] == "heavytail",
         "with the largest discrepancy in the scale mixture"),
        (all((d["target"] < 0) == (d["family"] in ("gauss", "lowkurt")) for d in FAM["level"].values()),
         "$K=pI_p$ and $D_p=-(p+3)G$"),
        (sum(abs(d["z"]) > 1.96 for d in FAM["rate"].values()) == 1 and abs(FAM["rate"]["heavytail_n1000"]["z"]) > 1.96
         and all(abs(d["z"]) <= 1.96 for d in FAM["level"].values()), "of the twelve points below"),
        (all(max(d["se"] for d in FAM["rate"].values() if d["family"] == f) / min(d["se"] for d in FAM["rate"].values() if d["family"] == f) < 1.1
             for f in ("gauss", "lowkurt", "heavytail", "misspec")), "remain roughly constant as $n$ grows"),
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
        (all(BA[k]["coverage"][m] >= 0.93 for k in ("gauss_low", "misspec_null", "skew_null", "heavy_low", "ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small") for m in ("IREG_HC1c", "IREG_HC3c", "REG_HC1", "REG_HC3")),
         "every procedure covers at least $93\\%$ at the $95\\%$ level"),
        (all(BA[k]["plug_over_Gn"] >= 4 for k in ("heavy_low", "ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small")) and BA["ln20_small"]["calibration"]["0.95"]["IREG_HC1o"] < 0.90,
         "variance inflation rather than estimation of a substantial population"),
        (all(BA[k]["median_width"]["IREG_HC1c"] >= BA[k]["median_width"]["REG_HC1"] for k in ("gauss_low", "misspec_null", "skew_null", "heavy_low", "ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small"))
         and all(BA[k]["median_width"]["IREG_HC3c"] >= BA[k]["median_width"]["REG_HC3"] for k in ("gauss_low", "misspec_null", "skew_null", "heavy_low", "ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small"))
         and all(BA[k]["coverage"]["IREG_HC3c"] > 0.95 for k in ("skew_null", "heavy_low", "ln12_small", "ln15_small", "heavy_ex", "ln20_small"))
         and all(BA[k]["mse_ratio"] >= 0.999 for k in ("gauss_low", "misspec_null", "skew_null", "heavy_low", "ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small")), "REG reaches the same coverage with shorter intervals"),
        (BA["gauss_low"]["plug_over_Gn"] < 2 and abs(BA["gauss_low"]["median_width"]["IREG_HC1c"] / BA["gauss_low"]["median_width"]["REG_HC1"] - 1) < 0.01,
         "the two estimators' intervals have the same width"),
        (all(BA[k]["calibration"]["0.80"]["IREG_HC1c"] < 0.79 and BA[k]["calibration"]["0.99"]["IREG_HC1c"] >= 0.989
             and all(BA[k]["calibration"][L]["IREG_HC1o"] < float(L) - 0.005 for L in ("0.80", "0.95", "0.99"))
             for k in ("ln15_small", "heavy_ex", "ln20_small")), "miscalibrated across nominal levels"),
        (all(abs(BA[k]["calibration"]["0.80"]["IREG_HC3"] - 0.80) < abs(BA[k]["calibration"]["0.80"]["IREG_HC1c"] - 0.80) for k in ("ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small")),
         "the plug-in correction has coverage between"),
        (C[g0]["calibration"]["0.80"]["REG_HC1"]["coverage"] < 0.80 and C[g0]["calibration"]["0.99"]["REG_HC1"]["coverage"] > 0.99
         and C[g0]["studentized"]["REG_HC1"]["levels"]["0.80"]["quantile_ratio"] > 1 > C[g0]["studentized"]["REG_HC1"]["levels"]["0.99"]["quantile_ratio"]
         and abs(C[g0]["calibration"]["0.80"]["IREG_HC3"]["coverage"] - 0.80) < abs(C[g0]["calibration"]["0.80"]["REG_HC1"]["coverage"] - 0.80),
         "REG+HC1 shows a similar, but smaller, pattern"),
        (C[g0]["calibration"]["0.80"]["REG_HC1"]["coverage"] < 0.80 and C[g0]["calibration"]["0.99"]["REG_HC1"]["coverage"] > 0.99,
         "is partly a crossing point"),
        (all(abs(v["frac_leverage_one_IREG"] - lev_th) < 3 * (lev_th * (1 - lev_th) / v["reps"]) ** 0.5
             and abs(v["frac_rank_deficient_IREG"] - rd_th) < 3 * (rd_th * (1 - rd_th) / v["reps"]) ** 0.5
             for v in R("r_gamma0_rare.json").values()), "the exact probabilities are"),
        # Section 3.3 connection and Appendix C tail sensitivity
        (all(abs(1 + population_(s_, 1.0, 0.0)["Delta"] / 500 - C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] / C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"]) < 0.015
             for s_ in (0.8,)), "versus a simulated ratio of"),
        (all(C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] > C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"]
             and population_(s_, 1.0, 0.0)["Delta"] > 0 for s_ in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)), "remains above one in"),
        (all(10 <= population_(sx, c_, g)["Delta"] / population_(sx, c_, g, 1 - 1e-4)["Delta"] <= 1000
             for sx, c_, g in ((1.8, 1.5, 0.3), (1.5, 1.0, 0.4), (2.0, 1.0, 0.0))), "by one to three orders of magnitude"),
        (all(1 + population_(sx, c_, g, 1 - 1e-8)["Delta"] / 500 >= 3 * sim for sx, c_, g, sim in
             ((1.8, 1.5, 0.3, hr["500"]["sim"]), (1.5, 1.0, 0.4, ex["sweep"]["1.5"]["500"]["sim"]),
              (2.0, 1.0, 0.0, C[g0]["sampling"]["IREG"]["mse"] / C[g0]["sampling"]["REG"]["mse"]))),
         "predictions several times larger than the simulated ratios"),
    ]:
        checks.append(("ok" if ok else "FAILED", text))
        expected[text] = lambda v: v == "ok"
    import run_validation_families as RVF
    checks += [
        # Appendix B, covariate families: the DGP constants quoted in the text, against the script that ran them
        (f"{FAM['level']['gauss_p2']['G']:.2f}", "with $G=0.49$"),
        (f"{FAM['level']['lowkurt_p2']['G']:.2f}", "($1.96$ for the Rademacher family)"),
        (f"{100 * RVF.EPS:.0f}", "where $2\\%$ of units have"),
        (f"{RVF.C_MIS}", "where $c=0.3$ and"),
        (f"{RVF.RHO}", "\\Sigma_{jk}=0.3^{|j-k|}$"),
        # Appendix B, CRT check: the predicted gap in the Gaussian design
        (f"{crt['gauss3_homo_n200']['predicted']:.0f}", "predicted $4G=5$"),
        # Appendix B, Results: the numbers quoted from Table 5
        (f"{full['est']['200']['cv']:.2f}", "estimate is $32.40$ at $n=200$"),
        (f"{full['formula']:.2f}", "$44.52$, a difference of"),
        (f"{full['formula'] - full['est']['200']['cv']:.1f}", "a difference of $12.1$"),
        (f"{abs(z200['cexp2_full']):.0f}", "(about $22$ standard errors)"),
        (f"{full['formula'] - full['est']['800']['cv']:.1f}", "falls to $5.7$ at $n=800$"),
        (f"{full['formula'] - full['est']['3200']['cv']:.2f}", "$2.55$ at $n=3200$"),
        (f"{max(abs(z) for z in z3200.values()):.1f}", "within $2.6$ standard errors of their"),
        (f"{z3200['cexp2_full']:+.1f}", "$z=-2.6$ in the all-blocks"),
        (f"{z3200['cexp2_het_null']:+.1f}", "$z=+2.3$ in the $\\gamma=0$"),
        (f"{va['cexp2_het_null']['est']['200']['cv']:.2f}", "$29.82$ at $n=200$"),
        (f"{va['cexp2_het_null']['est']['800']['cv']:.2f}", "$31.73$ at $n=800$"),
        (f"{va['cexp2_het_null']['est']['3200']['cv']:.2f}", "$33.25$ at $n=3200$"),
        # Appendix B, power: the raw estimator's standard errors
        (f"{min(v['est']['3200']['se_raw'] for v in va.values() if v['G'] > 0):.1f}", "$1.5$ to $3.8$"),
        (f"{max(v['est']['3200']['se_raw'] for v in va.values()):.1f}", "$1.5$ to $3.8$"),
        # Section 3.5: the s = 2 design of Table 4, plug-in against oracle
        (f"{100 * BA['ln20_small']['calibration']['0.80']['IREG_HC1c']:.1f}", "coverage is $75.1\\%$"),
        (f"{100 * BA['ln20_small']['calibration']['0.95']['IREG_HC1c']:.1f}", "$94.8\\%$, and $99.3\\%$"),
        (f"{100 * BA['ln20_small']['calibration']['0.99']['IREG_HC1c']:.1f}", "and $99.3\\%$ at nominal"),
        (f"{100 * BA['ln20_small']['calibration']['0.80']['IREG_HC1o']:.1f}", "the oracle gives $65.3\\%$"),
        (f"{100 * BA['ln20_small']['calibration']['0.95']['IREG_HC1o']:.1f}", "$83.5\\%$, and"),
        (f"{100 * BA['ln20_small']['calibration']['0.99']['IREG_HC1o']:.1f}", "$92.1\\%$."),
    ]
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
