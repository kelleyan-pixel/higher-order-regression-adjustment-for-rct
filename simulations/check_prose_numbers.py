"""Check the numbers and qualitative statements in the prose of paper/main.tex against results/.

Numbers in Section 3, the Discussion and parts of the appendices are generated macros
(paper/tables/gamma0_values.tex, written by make_tables.py) and cannot drift.  The remaining prose
numbers (introduction, Appendices B-D) are typed in the source; this script recomputes each of them
from results/*.json and asserts that (i) the recomputed, rounded value equals the value in the text
and (ii) that text actually occurs in paper/main.tex.  Qualitative statements are checked the same way:
each is tied to a test on the stored results and to the exact words it supports.

Usage: python3 simulations/check_prose_numbers.py
"""
import json, os, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_example import population as population_

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
import re as _re
# Compare against the typeset text only: drop LaTeX comments (an unescaped % to the end of the line), so that
# sentences surviving only in commented-out drafts cannot satisfy a check.
TEX = "\n".join(_re.sub(r"(?<!\\)%.*$", "", line)
                 for line in open(os.path.join(ROOT, "paper", "main.tex")).read().split("\n"))

SMALL = ["gauss_low", "misspec_null", "skew_null", "heavy_low", "ln12_small", "ln15_small", "ln15_g05", "heavy_ex",
         "ln20_small"]
SKEWED_SMALL = ["heavy_low", "ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small"]
SKEWED_HTE = ["ln15_mid", "heavy_high", "ln15_large"]
LV = ("0.80", "0.95", "0.99")


def main():
    ex, va, ab, crt, bi = R("example.json"), R("validation.json"), R("ablation.json"), \
        R("crt_correction.json"), R("bias.json")
    hr = ex["heavy_tail_case"]
    B = math.sqrt(400000)
    g3 = va["gauss3_homo"]["est"]
    full = va["cexp2_full"]
    z3200 = {k: (v["est"]["3200"]["cv"] - v["formula"]) / v["est"]["3200"]["se_cv"] for k, v in va.items()}
    z200 = {k: (v["est"]["200"]["cv"] - v["formula"]) / v["est"]["200"]["se_cv"] for k, v in va.items()}
    b800 = [abs(v[f"z_{e}"]) for k, v in bi.items() if k.endswith("n800") for e in ("REG", "IREG")]
    bf = bi["cexp2_full_n200"]
    BA = R("hte_battery.json")["designs"]
    FAM = R("validation_families.json")
    C = R("correction_summary.json")["designs"]
    g0 = "lognormal_s2.0_prop"
    sweep = ex["sweep"]
    cal = lambda k, L, m: BA[k]["calibration"][L][m]
    checks = [
        # (recomputed string, exact text that must appear in main.tex)
        # --- introduction
        # Introduction, example (1): "interacted regression has X% higher mean-squared error ... than simple
        # regression" means MSE_IREG / MSE_REG - 1 (not 1 - MSE_REG / MSE_IREG, which is 6.3%).
        (f"{100 * (ex['intro_ratio']['500']['sim'] - 1):.0f}\\%", "interacted regression has 7\\% higher mean-squared error"),
        # the design of example (1) is the one simulated by run_example.py: log X ~ N(0, 1.1^2), sigma(X) = 1 + X, slope 0.9
        (f"{math.sqrt(math.log(1 + ex['intro_population']['sd'] ** 2 / ex['intro_population']['mu'] ** 2)):.1f}^2",
         "\\log(X_i) \\sim \\nn\\p{0, \\, 1.1^2}"),
        # --- Appendix C
        (f"{hr['500']['sim']:.2f}", "simulated MSE ratios of $1.68$"),
        (f"{hr['2000']['sim']:.2f}", "$1.68$, $1.34$, and"),
        (f"{hr['8000']['sim']:.2f}", "$1.34$, and $1.19$"),
        (f"{hr['500']['predicted']:.1f}", "prediction is then $78.5$"),
        (f"{hr['2000']['predicted']:.1f}", "$78.5$, $20.4$, and"),
        (f"{hr['8000']['predicted']:.1f}", "and $5.8$"),
        (f"{hr['population']['R2tau']:.3f}", "R_\\tau^2\\approx0.009"),
        (f"{hr['population']['Delta'] / 1e4:.1f}", "3.9\\times10^4"),
        (f"{100 * hr['coverage']['IREG+HC1'][0]:.1f}", "without the plug-in correction is $85.6\\%$ with HC1"),
        (f"{100 * hr['coverage']['IREG+HC2'][0]:.1f}", "$88.9\\%$ with HC2"),
        (f"{100 * hr['coverage']['IREG+HC3'][0]:.1f}", "$94.0\\%$ with HC3"),
        (f"{100 * hr['coverage']['REG+HC1'][0]:.1f}", "$94.4\\%$ for REG+HC1"),
        (f"{100 * hr['coverage']['REG+HC3'][0]:.1f}", "$96.7\\%$ for REG+HC3"),
        (f"{sweep['1.5']['500']['predicted'] / sweep['1.5']['500']['sim']:.0f}", "the predicted ratio is four times the simulated ratio"),
        (f"{sweep['0.8']['skew']:.1f}", "For $s=0.8$ (skewness $3.7$)"),
        # --- Appendix B
        (f"{g3['200']['se_raw'] * B:.0f}", "from $151$"),
        (f"{g3['800']['se_raw'] * B:.0f}", "to $293$"),
        (f"{g3['200']['se_cv'] * B:.0f}", "is $56$"),
        (f"{g3['800']['se_cv'] * B:.0f}", "and $55$"),
        (f"{g3['3200']['se_raw'] / g3['3200']['se_cv']:.0f}", "ten times smaller"),
        (f"{full['formula'] - full['est']['800']['cv']:.1f}", "to $5.7$"),
        (f"{4 * full['G']:.1f}", "D_\\Bern=D_p-3.2"),
        (f"{ab['gauss3_homo'][[k for k in ab['gauss3_homo'] if 'Bern' in k][0]]:.0f}", "as many as thirty standard errors"),
        (f"{crt['gauss3_homo_n200']['diff']:.1f}\\pm{crt['gauss3_homo_n200']['se']:.1f}", "$4.3\\pm0.6$"),
        (f"{crt['gauss3_homo_n800']['diff']:.1f}\\pm{crt['gauss3_homo_n800']['se']:.1f}", "$6.3\\pm1.6$"),
        (f"{crt['cexp2_homo_n200']['diff']:.1f}\\pm{crt['cexp2_homo_n200']['se']:.1f}", "$3.0\\pm1.1$"),
        (f"{crt['cexp2_homo_n800']['diff']:.1f}\\pm{crt['cexp2_homo_n800']['se']:.1f}", "$8.6\\pm3.0$"),
        (f"{crt['gauss3_homo_n200']['predicted']:.0f}", "predicted $4G=5$"),
        (f"{max(b800):.1f}", "within $0.8$ Monte Carlo"),
        (f"{100 * (bf['bias_REG'] - bf['pred_REG']) / abs(bf['pred_REG']):.0f}", "about $11\\%$"),
        (f"{bf['z_REG']:.1f}", "($z=9.3$)"),
        (f"{100 * (bf['bias_IREG'] - bf['pred_IREG']) / abs(bf['pred_IREG']):.0f}", "by $5\\%$ of the predicted IREG"),
        (f"{bf['z_IREG']:.1f}", "bias ($z=3.5$)"),
        (f"{bi['cexp2_full_n800']['z_REG']:.1f}", "($z=0.4$ and"),
        (f"{bi['cexp2_full_n800']['z_IREG']:.1f}", "and $z=-0.5$)"),
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
        (f"{min(v['est']['3200']['se_raw'] for v in va.values() if v['G'] > 0):.1f}", "$1.5$ to $3.8$"),
        (f"{max(v['est']['3200']['se_raw'] for v in va.values()):.1f}", "$1.5$ to $3.8$"),
        (f"{FAM['level']['gauss_p2']['G']:.2f}", "with $G=0.49$"),
        (f"{FAM['level']['lowkurt_p2']['G']:.2f}", "($1.96$ for the Rademacher family)"),
    ]
    import run_validation_families as RVF
    checks += [
        (f"{100 * RVF.EPS:.0f}", "where $2\\%$ of units have"),
        (f"{RVF.C_MIS}", "where $c=0.3$ and"),
        (f"{RVF.RHO}", "\\Sigma_{jk}=0.3^{|j-k|}$"),
    ]
    expected = {  # the value each text snippet asserts, where it is not the number itself
        "the predicted ratio is four times the simulated ratio": lambda v: round(float(v)) == 4,
        "ten times smaller": lambda v: round(float(v)) == 10,
        "as many as thirty standard errors": lambda v: 28 <= float(v) <= 32,
        "3.9\\times10^4": lambda v: v == "3.9",
    }
    # Appendix D: probabilities in the rare-value design (250 units per arm, rho = 0.02)
    rho, m = 0.02, 250
    p0, p1 = (1 - rho) ** m, m * rho * (1 - rho) ** (m - 1)
    ok_rare = abs(100 * p0 - 0.6) < 0.05 and abs(100 * p1 - 3.3) < 0.1
    inc1 = full["est"]["800"]["cv"] - full["est"]["200"]["cv"]
    inc2 = full["est"]["3200"]["cv"] - full["est"]["800"]["cv"]
    W = lambda k, m: BA[k]["median_width"][m]
    qual = [
        # --- Section 3.1: jackknife implementation and cross-check of the NumPy engine
        (R("jackknife_check.json")["max_rel_gap"] < 1e-10 and R("jackknife_check.json")["omitted_loo_fits"] > 0,
         "The jackknife uses the exact deletion formula for least squares"),
        (max(R("r_crosscheck.json")["max_rel_gap"].values()) < 1e-8, "confirms agreement of estimates, standard errors, degrees of freedom"),
        (all(BA[k]["plug_mean"] > 0 for k in BA) and all(BA[k]["plug_over_Gn"] > 1 for k in BA if BA[k]["plug_over_Gn"] is not None),
         "is biased upward"),
        # --- Section 3.2
        (all(abs(BA[k]["mse_ratio"] - 1) < 3 * BA[k]["mse_ratio_mcse"] or BA[k]["mse_ratio"] > 1 for k in SMALL),
         "REG is also at least as precise a point estimator in these designs"),
        (all(BA[k]["coverage"]["IREG_HC1"] < BA[k]["coverage"]["IREG_HC1c"] - 0.01 and BA[k]["plug_over_Gn"] > 4 for k in SKEWED_SMALL),
         "the upward bias of the plug-in partly offsets"),
        (all(BA[k]["coverage"]["REG_JK"] > 0.96 for k in ("ln15_small", "heavy_ex", "ln20_small")),
         "under the heaviest tails it is conservative"),
        (all(BA[k]["coverage"][m] < 0.94 for k in ("ln15_mid", "ln15_large") for m in ("REG_HC1", "REG_JK"))
         and all(BA[k]["coverage"]["IREG_JK"] >= max(BA[k]["coverage"][m] for m in ("IREG_HC1c", "REG_HC1", "REG_JK")) for k in SKEWED_HTE),
         "REG undercovers with either variance estimator"),
        (all(BA[k]["coverage"]["IREG_JK"] >= max(BA[k]["coverage"][m] for m in ("IREG_HC1c", "REG_HC1", "REG_JK")) for k in SKEWED_HTE),
         "IREG with jackknife is the most reliable procedure"),
        (BA["rare"]["jackknife"]["REG"]["frac_reps_degenerate_loo"] < 0.001, "REG is essentially unaffected"),
        (all(abs(cal(k, "0.95", m) - 0.95) <= 0.02 and cal(k, "0.99", m) >= 0.983 for k in SMALL for m in ("IREG_HC1c", "REG_HC1"))
         and all(cal(k, "0.80", m) < 0.79 for k in ("ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small") for m in ("IREG_HC1c", "REG_HC1")),
         "HC1 intervals are close to nominal at $95\\%$ and $99\\%$ but undercover at $80\\%$ in the skewed designs"),
        (max(L_ - cal(k, f"{L_:.2f}", m) for k in SMALL for m in ("IREG_JK", "REG_JK") for L_ in (0.80, 0.95, 0.99))
         <= 2 * math.sqrt(0.8 * 0.2 / 20000),
         "With the jackknife, neither estimator is more than"),
        (max(L_ - cal(k, f"{L_:.2f}", "REG_JK") for k in SMALL for L_ in (0.80, 0.95, 0.99)) <= 2 * math.sqrt(0.8 * 0.2 / 20000),
         "consistently achieves at least nominal coverage"),
        (BA["ln15_large"]["coverage"]["IREG_JK"] < 0.92 and BA["heavy_high"]["coverage"]["REG_JK"] < 0.94,
         "REG undercovers with either variance estimator"),
        (max(BA["ln15_large"]["coverage"][m] for m in ("IREG_HC1c", "IREG_JK", "REG_HC1", "REG_JK")) < 0.93,
         "although it also reaches only"),
        (all(abs(cal(k, "0.95", m) - 0.95) <= 0.02 for k in SMALL for m in ("IREG_HC1c", "REG_HC1"))
         and min(cal(k, "0.80", "IREG_HC1c") for k in SMALL) < 0.76,
         "the undercoverage shows up more at other nominal levels"),
        (all(abs(cal(k, "0.95", m) - 0.95) <= 0.02 for k in SMALL for m in ("IREG_HC1c", "REG_HC1")),
         "HC1 intervals are close to nominal at $95\\%$ and $99\\%$"),
        (all(cal(k, "0.99", m) >= 0.983 for k in SMALL for m in ("IREG_HC1c", "REG_HC1")), "is partly a crossing point"),
        (all(BA[k]["group"] == "small" for k in SMALL) and all(BA[k]["R2_tau"] < 0.05 for k in SMALL), "small ($<0.05$)"),
        # --- Appendix B
        (1.5 <= inc1 / inc2 <= 2.5, "shrink by a\nfactor of about two"),
        (all(v["est"]["3200"]["se_raw"] / v["est"]["3200"]["se_cv"] >= 3.8 for v in va.values() if v["G"] > 0),
         "four to ten times larger in the heterogeneous designs"),
        (all(abs(v["est"]["3200"]["raw"] - (v["formula"] - 4 * v["G"])) <= 3 * v["est"]["3200"]["se_raw"] for v in va.values()),
         "($|z|\\leq3$ throughout)"),
        (RVF.KSC == 6.0, "have six times the standard deviation"),
        (abs(full["est"]["3200"]["cv"] - (full["formula"] - 4 * full["G"])) < full["est"]["3200"]["se_cv"],
         "standard error of the $n=3200$ estimate"),
        (max(FAM["level"].values(), key=lambda d: abs(d["mc_matrix_check"] - d["target"]) / abs(d["target"]))["family"] == "heavytail",
         "with the largest discrepancy in the scale mixture"),
        (all((d["target"] < 0) == (d["family"] in ("gauss", "lowkurt")) for d in FAM["level"].values()),
         "$K=pI_p$ and $D_p=-(p+3)G$"),
        (sum(abs(d["z"]) > 1.96 for d in FAM["rate"].values()) == 1 and abs(FAM["rate"]["heavytail_n1000"]["z"]) > 1.96
         and all(abs(d["z"]) <= 1.96 for d in FAM["level"].values()), "of the twelve points below"),
        (all(max(d["se"] for d in FAM["rate"].values() if d["family"] == f) / min(d["se"] for d in FAM["rate"].values() if d["family"] == f) < 1.1
             for f in ("gauss", "lowkurt", "heavytail", "misspec")), "remain roughly constant as $n$ grows"),
        # --- Appendix C
        (all(v[n]["sim"] > 1 and v[n]["predicted"] > 1 for v in sweep.values() for n in ("500", "2000", "8000"))
         and hr["500"]["sim"] > 1 and hr["500"]["predicted"] > 1,
         "agrees with the second-order calculation in every"),
        (all((v["formula"] > 0) == (v["est"]["3200"]["cv"] > 0) for v in va.values())
         and all((population_(s_, 1.0, 0.0)["Delta"] > 0) == (C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["IREG"]["mse"] > C[f"lognormal_s{s_:.1f}_prop"]["sampling"]["REG"]["mse"])
                 for s_ in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)),
         "The sign of the REG--IREG comparison"),
        (abs(C["lognormal_s0.6_prop"]["sampling"]["REG"]["sd"] / (population_(0.6, 1.0, 0.0)["V"] / 500) ** 0.5 - 1) < 0.02
         and C[g0]["sampling"]["REG"]["sd"] < 0.6 * (population_(2.0, 1.0, 0.0)["V"] / 500) ** 0.5,
         "is close to the first-order benchmark $\\sqrt{V^*/n}$"),
        (all(10 <= population_(sx, c_, g)["Delta"] / population_(sx, c_, g, 1 - 1e-4)["Delta"] <= 1000
             for sx, c_, g in ((1.8, 1.5, 0.3), (1.5, 1.0, 0.4), (2.0, 1.0, 0.0))), "by one to three orders of magnitude"),
        (all(1 + population_(sx, c_, g, 1 - 1e-8)["Delta"] / 500 >= 3 * sim for sx, c_, g, sim in
             ((1.8, 1.5, 0.3, hr["500"]["sim"]), (1.5, 1.0, 0.4, sweep["1.5"]["500"]["sim"]),
              (2.0, 1.0, 0.0, C[g0]["sampling"]["IREG"]["mse"] / C[g0]["sampling"]["REG"]["mse"]))),
         "predictions several times larger than the simulated ratios"),
        # --- Appendix D
        (ok_rare, "with probability about $0.6\\%$"),
        (ok_rare, "with probability about $3\\%$"),
    ]
    for ok, text in qual:
        checks.append(("ok" if ok else "FAILED", text))
        expected[text] = lambda v: v == "ok"
    implementation_anchors = {
        "The jackknife uses the exact deletion formula for least squares",
        "confirms agreement of estimates, standard errors, degrees of freedom",
    }
    verification = open(os.path.join(ROOT, "VERIFICATION.md")).read()
    bad = []
    for val, text in checks:
        document = verification if text in implementation_anchors else TEX
        if text not in document:
            bad.append(f"text not found in main.tex: {text!r}")
            continue
        ok = expected[text](val) if text in expected else (val.lstrip("+") in text.replace("{,}", ","))
        if not ok:
            bad.append(f"{text!r}: recomputed {val}")
    print(f"checked {len(checks)} prose statements; {len(bad)} problems")
    for b in bad:
        print("  ", b)
    assert not bad


if __name__ == "__main__":
    main()
