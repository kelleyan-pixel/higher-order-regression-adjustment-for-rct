"""Generate the LaTeX table bodies of the paper from results/*.json.

Every numeric entry in the paper's tables is written by this script, so the
tables cannot drift from the stored simulation output.

Usage: python make_tables.py          (writes paper/tables/*.tex)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "paper", "tables")
load = lambda f: json.load(open(os.path.join(RES, f)))

LAB1_UNUSED = {"gauss_iid": "Gaussian, iid",
        "gauss_indep_arms": "Gaussian, independent arm noise",
        "skew_het": "Skewed + heteroskedastic",
        "null_hte": "Null HTE",
        "heavy_het": "Heavy-tailed + heteroskedastic",
        "rare_extreme": "Rare extreme",
        "large_hte": "Large HTE ($\\gamma^*=3$)"}
LAB3 = {"gauss_iid": "Gaussian, iid", "skew_het": "Skewed + het.",
        "misspec": "Nonlinear misspec.", "large_hte": "Large HTE"}
LAB_V = {"gauss3_homo": "Gaussian, homoskedastic, $p=3$ ($F$)",
         "cexp2_homo": "Exponential covariates, homoskedastic ($F,Q,R$)",
         "cexp2_het_null": "Exponential, heteroskedastic, $\\gamma=0$",
         "cexp2_full": "Exponential, HTE + arm-specific misspec.\\ (all)",
         "gauss2_misspec_het": "Gaussian, arm-specific misspec.\\ ($\\tr E^2,E_W$)",
         "mixed2_het": "Mixed skewed/Gaussian, heteroskedastic"}


def write(name, lines):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name, f"({len(lines)} rows)")


def validation():
    r = load("validation.json")
    rows = []
    for k, lab in LAB_V.items():
        v = r[k]
        cells = " & ".join(f"${v['est'][str(n)]['cv']:.2f}\\ ({v['est'][str(n)]['se_cv']:.2f})$"
                           for n in [200, 800, 3200])
        rows.append(f"{lab} & ${v['formula']:.2f}$ & {cells} \\\\")
    write("validation_body.tex", rows)


def ablation():
    r = load("ablation.json")
    labs = [k for k in next(iter(r.values())) if not k.startswith("_")]
    rows = [f"{LAB_V.get(k, k)} & " + " & ".join(f"${v[l]:+.1f}$" for l in labs) + " \\\\"
            for k, v in r.items()]
    write("ablation_body.tex", rows)


def validity():
    r = load("example.json")["sweep"]
    rows = []
    for s, row in r.items():
        cells = " & ".join(f"{row[str(n)]['sim']:.4f}\\,({1e4 * row[str(n)]['se']:.0f}) & {row[str(n)]['predicted']:.4f}"
                           for n in [500, 2000, 8000])
        rows.append(f"{float(s):.1f} & {row['skew']:.1f} & {row['Delta']:.0f} & {cells} \\\\")
    write("validity_body.tex", rows)


# ---------------------------------------------------------------- gamma = 0 (R)
from math import exp, sqrt
from scipy.stats import norm

GROUPS = ["skew", "misspec", "nsweep", "p5", "rare"]


def load_r():
    r = {}
    for g in GROUPS:
        r.update(load(f"r_gamma0_{g}.json"))
    r2 = load("r_gamma0_rare_estimatr2.json")
    return r, r2


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def cov(v, est, cat="all"):
    return v[est][cat]["coverage"]


def gamma0_tables():
    r, r2 = load_r()
    vals = {}
    # --- skewness sweep
    CS = load("correction_summary.json")["designs"]
    fm = lambda x: f"{x:.4f}" if x < 1 else (f"{x:.3f}" if x < 10 else f"{x:.2f}")
    rows = []
    for s_ in [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
        v = r[f"lognormal_s{s_:.1f}_prop"]; d = CS[f"lognormal_s{s_:.1f}_prop"]
        wd, sm = d["median_width_95"], d["sampling"]
        skew = (exp(s_ ** 2) + 2) * sqrt(exp(s_ ** 2) - 1)
        cells = []
        for est in ("IREG", "REG"):
            cells.append(f"{sm[est]['mse'] ** 0.5:.3f}")
            for hc in ("HC1", "HC3"):
                cells += [f"{wd[f'{est}_{hc}']:.2f}", pct(cov(v, f"{est}_{hc}"))]
        rows.append(f"{s_:.1f} & {skew:.1f} & " + " & ".join(cells) + " \\\\")
    write("null_main_body.tex", rows)
    # --- other designs
    lab = {"gauss_quad": "Gaussian $X$, $0.3(X^2-1)$",
           "lognormal_s1.0_log": "$s=1$, $2\\log X$",
           "lognormal_s1.0_quadstd": "$s=1$, $0.3(Z^2-1)$",
           "lognormal_s1.0_log_prop": "$s=1$, $2\\log X$, $\\sigma=1+X$",
           "lognormal_s1.2_prop_n200": "$s=1.2$, $\\sigma=1+X$",
           "lognormal_s1.2_prop": "$s=1.2$, $\\sigma=1+X$",
           "lognormal_s1.2_prop_n2000": "$s=1.2$, $\\sigma=1+X$",
           "lognormal_s1.2_prop_n8000": "$s=1.2$, $\\sigma=1+X$",
           "lognormal_s1.0_prop_p5_n500": "$p=5$, $s=1$, $\\sigma=1+X_1$",
           "lognormal_s1.0_prop_p5_n200": "$p=5$, $s=1$, $\\sigma=1+X_1$"}
    blocks = [("Misspecification, homoskedastic unless noted", ["gauss_quad", "lognormal_s1.0_log",
               "lognormal_s1.0_quadstd", "lognormal_s1.0_log_prop"]),
              ("Sample size", ["lognormal_s1.2_prop_n200", "lognormal_s1.2_prop",
               "lognormal_s1.2_prop_n2000", "lognormal_s1.2_prop_n8000"]),
              ("More covariates", ["lognormal_s1.0_prop_p5_n500", "lognormal_s1.0_prop_p5_n200"])]
    rows = []
    for j, (title, keys) in enumerate(blocks):
        if j:
            rows.append("\\addlinespace")
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for k in keys:
            v = r[k]
            n = v["design"]["n"]
            cells = [cov(v, e) for e in ["IREG_HC1", "IREG_HC3", "REG_HC1", "REG_HC3"]]
            rows.append(f"\\quad {lab[k]} & {n:,} & {v['reps']:,} & ".replace(",", "{,}")
                        + " & ".join(pct(c) for c in cells) + " \\\\")
    write("gamma0_other_body.tex", rows)
    # --- rare value / leverage one, estimatr 1.0.2 and 2.0.0
    rows = []
    for key, title in [("rare_q0.02_prop", "$\\sigma = 1+|X_1|$"), ("rare_q0.02_homo", "$\\sigma = 1$")]:
        v, w = r[key], r2[key]
        # REG never isolates a unit, so it cannot depend on the version; IREG+HC1 agrees except in
        # rank-deficient replications, where the versions drop different collinear columns.
        for e in ["REG_HC1", "REG_HC3"]:
            for c in ["all", "full_rank", "leverage_one"]:
                assert v[e][c]["coverage"] == w[e][c]["coverage"], (key, e, c)
        for c in ["full_rank", "leverage_one"]:
            assert v["IREG_HC1"][c]["coverage"] == w["IREG_HC1"][c]["coverage"], (key, c)
        rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{Rare value, {title}}}}} \\\\")
        for c, clab in [("all", "all replications"), ("full_rank", "full-rank designs"),
                        ("leverage_one", "\\quad of which leverage-one")]:
            cells = [cov(v, "IREG_HC1", c), cov(v, "IREG_HC2", c), cov(v, "IREG_HC3", c),
                     cov(w, "IREG_HC1", c), cov(w, "IREG_HC2", c), cov(w, "IREG_HC3", c),
                     cov(v, "REG_HC1", c), cov(v, "REG_HC3", c)]
            rows.append(f"\\quad {clab} & " + " & ".join(pct(x) for x in cells) + " \\\\")
        rows.append("\\addlinespace")
    write("gamma0_rare_body.tex", rows[:-1])
    # --- values for the prose (\gz{key})
    for k, v in r.items():
        for e in ["IREG_HC1", "IREG_HC2", "IREG_HC3", "REG_HC1", "REG_HC3"]:
            for c, cl in [("all", "all"), ("full_rank", "full"), ("leverage_one", "lev1")]:
                vals[f"{k}:{e}:{cl}"] = pct(v[e][c]["coverage"])
                if v[e][c].get("frac_no_interval") is not None:
                    vals[f"{k}:{e}:{cl}:nointerval"] = pct(v[e][c]["frac_no_interval"])
        vals[f"{k}:rankdef"] = pct(v["frac_rank_deficient_IREG"])
        vals[f"{k}:lev1"] = pct(v["frac_leverage_one_IREG"])
        vals[f"{k}:medlev"] = f"{v['median_max_leverage_IREG']:.2f}"
        vals[f"{k}:reps"] = f"{v['reps']:,}".replace(",", "{,}")
    for k, v in r2.items():
        for e in ["IREG_HC1", "IREG_HC2", "IREG_HC3"]:
            for c, cl in [("all", "all"), ("full_rank", "full"), ("leverage_one", "lev1")]:
                vals[f"{k}@v2:{e}:{cl}"] = pct(v[e][c]["coverage"])
    reg = [cov(v, "REG_HC1") for v in r.values()]
    vals["all:REG_HC1:min"], vals["all:REG_HC1:max"] = pct(min(reg)), pct(max(reg))
    top = max(v["reps"] for v in r.values())             # the 20,000-replication designs
    mc = [v[e]["all"]["mcse"] for v in r.values() for e in ["IREG_HC1", "IREG_HC2", "IREG_HC3", "REG_HC1", "REG_HC3"]
          if v["reps"] == top]
    vals["mcse:max20k"] = f"{100 * max(mc):.2f}"
    vals["mcse:maxall"] = f"{100 * max(v[e]['all']['mcse'] for v in r.values() for e in ['IREG_HC1', 'IREG_HC2', 'IREG_HC3', 'REG_HC1', 'REG_HC3']):.2f}"
    vals["estimatr:v1"] = next(iter(r.values()))["estimatr_version"]
    vals["estimatr:v2"] = next(iter(r2.values()))["estimatr_version"]
    for R2 in [0.01, 0.1]:
        vals[f"analytic:cov:{R2}"] = f"{100 * (2 * norm.cdf(norm.ppf(0.975) * sqrt(1 - R2)) - 1):.1f}"
    # --- validation artefacts
    V = load("r_validation.json"); V2 = load("r_validation_estimatr2.json"); C = load("r_crosscheck.json")
    vals["val:datasets"] = str(V["datasets_checked"])
    vals["val:formulagap"] = sci(max(V['formula_vs_engine_max_rel_gap'], V2['formula_vs_engine_max_rel_gap']))
    vals["val:handgap"] = sci(max(V['hand_vs_estimatr_max_rel_gap'], V2['hand_vs_estimatr_max_rel_gap']))
    vals["val:crossgap"] = sci(max(C['max_rel_gap'].values()))
    vals["val:crossn"] = str(C["datasets"])
    L1, L2 = V["leverage_one"], V2["leverage_one"]
    vals["val:lev:onem"] = sci(L1['one_minus_leverage'])
    vals["val:lev:resid"] = sci(abs(L1['residual']))
    vals["val:lev:hctwo"] = f"{L1['se_with_unit_contribution_zeroed']['HC2']:.3f}"
    vals["val:lev:hcthree:zero"] = f"{L1['se_with_unit_contribution_zeroed']['HC3']:.3f}"
    vals["val:lev:hcthree:v1"] = f"{L1['estimatr_se']['HC3']:.3f}"
    vals["val:lev:hcthree:v2"] = f"{L2['estimatr_se']['HC3']:.3f}"
    vals["val:plugin"] = f"{V['gamma0']['plugin_deltaSigmadelta_over_n']:.4f}"
    vals["val:p1gap"] = sci(max(V['p1_closed_form_max_rel_gap'], V2['p1_closed_form_max_rel_gap']))
    vals.update(correction_tables())
    vals.update(diagnostic_tables())
    vals.update(hte_tables())
    vals.update(families_tables())
    lines = ["% generated by simulations/make_tables.py from results/*.json -- do not edit"]
    for k in sorted(vals):
        lines.append(f"\\expandafter\\def\\csname gz:{k}\\endcsname{{{vals[k]}}}")
    write("gamma0_values.tex", lines)


# ---------------------------------------------------------------- correction diagnostic
CAL_DESIGNS = [("lognormal_s2.0_prop", "$\\gamma=0$, skewed ($s=2$)"),
               ("heavy_tail_example", "Small $R^2_\\tau$, heavy-tailed example"),
               ("large_hte_n500_p5", "Large $R^2_\\tau$")]
CAL_METHODS = [("IREG_HC1", "HC1"), ("IREG_HC1c", "HC1 + plug-in"), ("IREG_HC1o", "HC1 + $G/n$ (oracle)"),
               ("IREG_HC3", "HC3"), ("IREG_HC3c", "HC3 + plug-in"), ("REG_HC1", "REG, HC1")]
LEVELS = ["0.80", "0.90", "0.95", "0.98", "0.99"]


def correction_tables():
    S = load("correction_summary.json")["designs"]
    rows, vals, csv_rows = [], {}, ["design,method,nominal,coverage,calibration_error,mcse"]
    for j, (d, title) in enumerate(CAL_DESIGNS):
        s = S[d]; c = s["calibration"]
        if j:
            rows.append("\\addlinespace")
        r2 = s["truth"]["R2_tau"]
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}: $R^2_\\tau={r2:.3f}$}} \\\\" if r2 > 0 else
                    f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}: $R^2_\\tau=0$ exactly}} \\\\")
        for m, mlab in CAL_METHODS:
            rows.append(f"\\quad {mlab} & " + " & ".join(f"{100 * c[l][m]['coverage']:.1f}" for l in LEVELS) + " \\\\")
            for l in LEVELS:
                cv, se = c[l][m]["coverage"], c[l][m]["mcse"]
                vals[f"corr:{d}:{m}:{l}"] = f"{100 * cv:.1f}"
                csv_rows.append(f"{d},{m},{l},{cv:.6f},{cv - float(l):+.6f},{se:.6f}")
        vals[f"corr:{d}:R2"] = f"{s['truth']['R2_tau']:.3f}"
        r = s["plugin"]["ratio_mean_to_true"]
        vals[f"corr:{d}:ratio"] = "--" if r is None else (f"{r:.2f}" if r < 10 else f"{r:.0f}")
        vals[f"corr:{d}:reps"] = f"{s['reps']:,}".replace(",", "{,}")
    # 95% summary with interval width (Table: tab:correction95).  The width statistic is the stored
    # median over replications of width(HC1 + plug-in) / width(HC1) (correction_common.summarize).
    rows95 = []
    for d, title in CAL_DESIGNS:
        s = S[d]; c = s["calibration"]["0.95"]; r2 = s["truth"]["R2_tau"]
        wr = s["width"]["HC1"]["median_width_ratio"]
        rows95.append(f"{title} & " + (f"{r2:.3f}" if r2 > 0 else "0") + " & "
                      + " & ".join(f"{100 * c[m]['coverage']:.1f}" for m in ("IREG_HC1", "IREG_HC1c", "IREG_HC1o"))
                      + f" & {wr:.2f} \\\\")
        vals[f"corr:{d}:wr"] = f"{wr:.2f}"
        vals[f"corr:{d}:wr_miss"] = f"{s['width']['HC1']['median_width_ratio_uncorrected_misses']:.2f}"
        vals[f"corr:{d}:wr_cover"] = f"{s['width']['HC1']['median_width_ratio_uncorrected_covers']:.2f}"
    write("correction95_body.tex", rows95)
    # Corollary 4: in the Gaussian, homoskedastic, well-specified large-heterogeneity design the
    # deficiency is exactly -(p+1) R2_tau observations.
    L = S["large_hte_n500_p5"]
    vals["corr:large_hte_n500_p5:Delta"] = f"{(L['p'] + 1) * L['truth']['R2_tau']:.1f}"
    mc = max(S[d]["calibration"][l][m]["mcse"] for d, _ in CAL_DESIGNS for l in LEVELS for m, _ in CAL_METHODS)
    vals["corr:mcse:max"] = f"{100 * mc:.1f}"
    write("calibration_body.tex", rows)
    open(os.path.join(RES, "calibration.csv"), "w").write("\n".join(csv_rows) + "\n")
    calibration_figure(S)
    return vals



def hte_tables():
    """Section 3.4-3.5: controlled HTE sweep and combined-design battery (results/hte_*.json)."""
    from scipy.stats import norm
    vals = {}
    SW, BA = load("hte_sweep.json"), load("hte_battery.json")
    M = ["IREG_HC1", "IREG_HC1c", "IREG_HC1o", "IREG_HC3", "IREG_HC3c", "REG_HC1", "REG_HC3"]
    for tag, R in (("sw", SW), ("ba", BA)):
        for k, d in R["designs"].items():
            vals[f"hte:{tag}:{k}:R2"] = f"{d['R2_tau']:.3f}" if d["R2_tau"] < 0.01 else f"{d['R2_tau']:.2f}"
            vals[f"hte:{tag}:{k}:mse"] = f"{d['mse_ratio']:.2f}"
            vals[f"hte:{tag}:{k}:gstar"] = f"{d['gamma_star']:.2f}"
            if d["plug_over_Gn"] is not None:
                vals[f"hte:{tag}:{k}:plugratio"] = f"{d['plug_over_Gn']:.2f}"
            for m in M:
                vals[f"hte:{tag}:{k}:{m}:cov"] = pct(d["coverage"][m])
                vals[f"hte:{tag}:{k}:{m}:width"] = f"{d['median_width'][m]:.2f}"
            vals[f"hte:{tag}:{k}:lev1"] = pct(d["frac_leverage_one"])
            vals[f"hte:{tag}:{k}:rankdef"] = pct(d["frac_rank_deficient"])
    vals["hte:reps"] = f"{SW['reps']:,}".replace(",", "{,}")
    vals["hte:mcse:max"] = f"{100 * max(d['mcse'][m] for R in (SW, BA) for d in R['designs'].values() for m in M):.2f}"
    # Table B: sweep
    rows = []
    for k, d in SW["designs"].items():
        c, w, r2 = d["coverage"], d["median_width"], d["R2_tau"]
        asy = 2 * norm.cdf(1.96 * sqrt(1 - r2)) - 1
        vals[f"hte:sw:{k}:asy"] = pct(asy)
        rows.append(f"{r2:.2f} & {pct(c['IREG_HC1'])} & {pct(asy)} & {pct(c['IREG_HC1c'])} & {pct(c['IREG_HC1o'])} & {pct(c['REG_HC1'])}"
                    f" & {w['IREG_HC1']:.3f} & {w['IREG_HC1c']:.3f} & {w['REG_HC1']:.3f} \\\\")
    write("hte_sweep_body.tex", rows)
    dev = max(abs(d["coverage"]["IREG_HC1c"] - d["coverage"]["IREG_HC1o"]) for d in SW["designs"].values())
    vals["hte:sw:maxdev:c:o"] = f"{100 * dev:.1f}"
    vals["hte:sw:reg:min"] = pct(min(d["coverage"]["REG_HC1"] for d in SW["designs"].values()))
    vals["hte:sw:reg:max"] = pct(max(d["coverage"]["REG_HC1"] for d in SW["designs"].values()))
    vals["hte:sw:oracle:min"] = pct(min(d["coverage"]["IREG_HC1o"] for d in SW["designs"].values()))
    vals["hte:sw:oracle:max"] = pct(max(d["coverage"]["IREG_HC1o"] for d in SW["designs"].values()))
    # Table C: battery, all four procedures; Table D: calibration of the plug-in with small R^2_tau
    fw = lambda w: f"{w:.2f}" if w < 10 else f"{w:.1f}"
    fr2 = lambda r: "0" if r == 0 else (f"{r:.3f}" if r < 0.01 else f"{r:.2f}")
    GROUPS = [("small", "Small or no heterogeneity"), ("moderate", "Moderate heterogeneity"), ("large", "Large heterogeneity")]
    seq = ["gauss_low", "misspec_null", "skew_null", "heavy_low", "ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small",
           "ln15_mid", "rare", "heavy_high", "ln15_large", "gauss_high"]
    order = [k for k in seq if k in BA["designs"]]
    METH = ("IREG_HC1", "IREG_HC1c", "IREG_HC1o", "IREG_HC3", "IREG_HC3c", "REG_HC1", "REG_HC3")
    rows, rows2, rows3 = [], [], []
    for gi, (g, title) in enumerate(GROUPS):
        if gi:
            rows.append("\\addlinespace")
        rows.append(f"\\multicolumn{{12}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for k in [k for k in order if BA["designs"][k]["group"] == g]:
            d = BA["designs"][k]; c, w, cal = d["coverage"], d["median_width"], d["calibration"]
            if k == "heavy_ex":
                d = dict(d, label="Lognormal $s=1.8$")
            cell = lambda m: f"{pct(c[m])} ({fw(w[m])})"
            cols = []
            for est, ms in (("IREG", ("IREG_HC1c", "IREG_HC3c")), ("REG", ("REG_HC1", "REG_HC3"))):
                cols.append(f"{d['mse_' + est] ** 0.5:.3f}")
                for m in ms:
                    cols += [fw(w[m]), pct(c[m])]
            rows.append(f"\\quad {d['label']} & {fr2(d['R2_tau'])} & " + " & ".join(cols) + " \\\\")
            pr = "--" if d["plug_over_Gn"] is None else (f"{d['plug_over_Gn']:.1f}" if d["plug_over_Gn"] < 100 else f"{d['plug_over_Gn']:.0f}")
            rows2.append(f"{d['label']} & {fr2(d['R2_tau'])} & {pct(c['IREG_HC1'])} & {pct(c['IREG_HC1o'])} & {pct(c['IREG_HC3'])} & {pr} \\\\")
            vals[f"hte:ba:{k}:plugratio:fmt"] = pr
            vals[f"hte:ba:{k}:wr1"] = f"{w['IREG_HC1c'] / w['REG_HC1']:.2f}"
            vals[f"hte:ba:{k}:wr3"] = f"{w['IREG_HC3c'] / w['REG_HC3']:.2f}"
            for m in METH:
                for L in cal:
                    vals[f"hte:ba:{k}:{m}:cal:{L}"] = pct(cal[L][m])
            if g == "small" and d["G"] > 0:
                lv = ("0.80", "0.95", "0.99")
                rows3.append(f"{d['label']} & {fr2(d['R2_tau'])} & {pr} & "
                             + " & ".join(pct(cal[L]["IREG_HC1o"]) for L in lv) + " & "
                             + " & ".join(pct(cal[L]["IREG_HC1c"]) for L in lv) + " & "
                             + " & ".join(pct(cal[L]["REG_HC1"]) for L in lv) + f" & {w['IREG_HC1c'] / w['REG_HC1']:.2f} \\\\")
    small = [BA["designs"][k] for k in order if BA["designs"][k]["group"] == "small"]
    smallG = [d for d in small if d["G"] > 0]
    for tag, m in (("I1c", "IREG_HC1c"), ("R1", "REG_HC1"), ("I1", "IREG_HC1"), ("I3c", "IREG_HC3c"), ("I3", "IREG_HC3")):
        xs = [d["coverage"][m] for d in small]
        vals[f"hte:ba:small:{tag}:min"], vals[f"hte:ba:small:{tag}:max"] = pct(min(xs)), pct(max(xs))
    vals["hte:ba:small:wr1:max"] = f"{max(d['median_width']['IREG_HC1c'] / d['median_width']['REG_HC1'] for d in small):.2f}"
    vals["hte:ba:small:wr3:max"] = f"{max(d['median_width']['IREG_HC3c'] / d['median_width']['REG_HC3'] for d in small):.2f}"
    vals["hte:ba:small:plug:min"] = f"{min(d['plug_over_Gn'] for d in smallG):.0f}"
    vals["hte:ba:small:plug:max"] = f"{max(d['plug_over_Gn'] for d in smallG):.0f}"
    smallNG = [d for d in smallG if not d["label"].startswith("Gaussian")]
    vals["hte:ba:smallng:plug:min"] = f"{min(d['plug_over_Gn'] for d in smallNG):.0f}"
    vals["hte:ba:smallng:plug:max"] = f"{max(d['plug_over_Gn'] for d in smallNG):.0f}"
    vals["hte:ba:ndesigns"] = str(len(BA["designs"]))
    x80 = [BA["designs"][k]["calibration"]["0.80"]["IREG_HC3"] for k in ("ln12_small", "ln15_small", "ln15_g05", "heavy_ex", "ln20_small")]
    vals["hte:ba:small:I3cal80:min"], vals["hte:ba:small:I3cal80:max"] = pct(min(x80)), pct(max(x80))
    vals["hte:ba:mse:min"] = f"{min(d['mse_ratio'] for d in BA['designs'].values()):.2f}"
    vals["hte:ba:mse:max"] = f"{max(d['mse_ratio'] for d in BA['designs'].values()):.2f}"
    write("hte_battery_body.tex", rows)
    write("hte_battery_supp_body.tex", rows2)
    write("hte_calibration_body.tex", rows3)
    hte_figure(SW)
    return vals


def hte_figure(SW):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    D = list(SW["designs"].values())
    r2 = [d["R2_tau"] for d in D]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    grid = np.linspace(0, 0.9, 200)
    ax.plot(grid, 100 * (2 * norm.cdf(1.96 * np.sqrt(1 - grid)) - 1), color="0.6", lw=1, ls=":", label="asymptotic, no correction")
    for m, lab, st in (("IREG_HC1", "IREG, HC1", dict(marker="o", color="C3")),
                       ("IREG_HC1c", "IREG, HC1 + plug-in", dict(marker="s", color="C0")),
                       ("IREG_HC1o", "IREG, HC1 + $G/n$ (oracle)", dict(marker="^", color="C2", ls="--")),
                       ("REG_HC1", "REG, HC1", dict(marker="D", color="0.2", ls="-."))):
        ax.plot(r2, [100 * d["coverage"][m] for d in D], label=lab, ms=4, lw=1.2, **st)
    ax.axhline(95, color="0.4", lw=0.8)
    ax.set_xlabel(r"$R^2_\tau$"); ax.set_ylabel("coverage of nominal 95% interval (%)")
    ax.set_ylim(40, 100); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "paper", "figures", "hte_coverage.pdf"), metadata={"CreationDate": None})
    plt.close(fig)


FAM_LAB = {"gauss": "Gaussian", "lowkurt": "Rademacher", "heavytail": "Scale mixture", "misspec": "Quadratic misspec."}
VAL_SHORT = {"gauss3_homo": "Gaussian, $p=3$", "cexp2_homo": "Exponential, homosk.", "cexp2_het_null": "Exponential, het., $\\gamma=0$",
             "cexp2_full": "Exponential, all blocks", "gauss2_misspec_het": "Gaussian, misspec.", "mixed2_het": "Mixed, het."}


def families_tables():
    """Appendix B: the covariate-family table and the two validation figures (results/validation_families.json)."""
    F, V = load("validation_families.json"), load("validation.json")
    vals, rows = {}, []
    for k, d in F["level"].items():
        rows.append(f"{FAM_LAB[d['family']]} & {d['p']} & ${d['target']:.2f}$ & ${d['est']:.2f}\\ ({d['se']:.2f})$ & ${d['z']:+.2f}$ & ${d['se_raw']:.2f}$ \\\\")
    write("validation_families_body.tex", rows)
    L = list(F["level"].values()); R = list(F["rate"].values())
    vals["fam:n"] = f"{F['n_level']:,}".replace(",", "{,}"); vals["fam:B"] = f"{F['B_level']:,}".replace(",", "{,}")
    vals["fam:maxabsz"] = f"{max(abs(d['z']) for d in L):.2f}"
    vals["fam:rate:maxabsz"] = f"{max(abs(d['z']) for d in R):.2f}"
    ratio = [d["se_raw"] / d["se"] for d in L]
    vals["fam:seratio:min"], vals["fam:seratio:max"] = f"{min(ratio):.1f}", f"{max(ratio):.1f}"
    vals["fam:matrixcheck:maxrel"] = f"{100 * max(abs(d['mc_matrix_check'] - d['target']) / abs(d['target']) for d in L):.0f}"
    vals["fam:target:min"], vals["fam:target:max"] = f"{min(d['target'] for d in L):.1f}", f"{max(d['target'] for d in L):.1f}"
    allz = [(v["est"]["3200"]["cv"] - v["formula"]) / v["est"]["3200"]["se_cv"] for v in V.values()] + [d["z"] for d in L]
    vals["valall:n"] = str(len(allz)); vals["valall:within196"] = str(sum(abs(z) <= 1.96 for z in allz))
    h = {d["n"]: d["z"] for d in R if d["family"] == "heavytail"}
    vals["fam:rate:heavytail1000:z"] = f"{h[1000]:+.2f}"
    vals["fam:rate:heavytail:nbr"] = f"{max(abs(h[500]), abs(h[2000])):.1f}"
    vals["valall:maxabsz"] = f"{max(abs(z) for z in allz):.2f}"
    validation_figures(F, V)
    return vals


def validation_figures(F, V):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    figdir = os.path.join(HERE, "..", "paper", "figures")
    tex = lambda t: t.replace("\\gamma", "\u03b3").replace("$", "")
    # z-statistics for every design, sorted by D_p
    items = [(v["formula"], tex(VAL_SHORT[k]), (v["est"]["3200"]["cv"] - v["formula"]) / v["est"]["3200"]["se_cv"], "C0") for k, v in V.items()]
    items += [(d["target"], f"{FAM_LAB[d['family']]}, p={d['p']}", d["z"], "C1") for d in F["level"].values()]
    items.sort(key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(range(len(items)), [t[2] for t in items], color=[t[3] for t in items], width=0.65)
    for y in (-1.96, 1.96):
        ax.axhline(y, color="C3", ls="--", lw=0.9)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([f"{t[1]}  ({t[0]:.1f})" for t in items], rotation=55, ha="right", fontsize=6.5)
    ax.set_ylabel(r"$z=(\hat D_p-D_p)/\widehat{\mathrm{SE}}$"); ax.set_ylim(-3.2, 3.2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="C0", label="designs with independent coordinates, p \u2264 3"), Patch(color="C1", label="covariate families, p = 2, 5, 8")],
              frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "validation_z.pdf"), metadata={"CreationDate": None}); plt.close(fig)
    # trajectories: estimate +- 1.96 SE against n, exact target dashed
    panels = [(tex(VAL_SHORT[k]), v["formula"], [(int(n), e["cv"], e["se_cv"]) for n, e in v["est"].items()]) for k, v in V.items()]
    for fam in ("gauss", "lowkurt", "heavytail", "misspec"):
        pts = [(d["n"], d["est"], d["se"]) for d in F["rate"].values() if d["family"] == fam]
        lv = F["level"][f"{fam}_p5"]; pts.append((F["n_level"], lv["est"], lv["se"]))
        panels.append((f"{FAM_LAB[fam]}, p=5", lv["target"], sorted(pts)))
    fig, axes = plt.subplots(2, 5, figsize=(9.5, 4.0))
    for ax, (title, target, pts) in zip(axes.ravel(), panels):
        nn, est, se = zip(*pts)
        ax.errorbar(nn, est, yerr=1.96 * np.array(se), fmt="o-", ms=3, lw=1, capsize=2, color="C0")
        ax.axhline(target, color="C3", ls="--", lw=0.9)
        ax.set_xscale("log"); ax.set_xticks(nn); ax.set_xticklabels([str(x) for x in nn], fontsize=6); ax.minorticks_off()
        ax.set_title(title, fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\hat D_p$")
    for ax in axes[1]:
        ax.set_xlabel("$n$")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "validation_rates.pdf"), metadata={"CreationDate": None}); plt.close(fig)

def calibration_figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.hashsalt": "calibration"})
    nominal = [100 * float(l) for l in LEVELS]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    styles = [("IREG_HC1", "HC1", dict(color="black", ls="-", marker="o", ms=3)),
              ("IREG_HC1c", "HC1 + plug-in", dict(color="tab:red", ls="--", marker="s", ms=3)),
              ("IREG_HC1o", "HC1 + $G/n$ (oracle)", dict(color="tab:blue", ls=":", marker="x", ms=5))]
    for ax, (d, title) in zip(axes, CAL_DESIGNS):
        c = S[d]["calibration"]
        ax.plot([50, 100], [50, 100], color="0.7", lw=0.8, zorder=0)
        for m, lab, kw in styles:
            ax.plot(nominal, [100 * c[l][m]["coverage"] for l in LEVELS], label=lab, lw=1, **kw)
        ax.set_title(title.replace("$R^2_\\tau$", r"$R^2_\tau$"), fontsize=8)
        ax.set_xlim(78, 100); ax.set_ylim(50, 100); ax.set_xlabel("nominal coverage (%)")
        ax.set_xticks([80, 90, 95, 99]); ax.set_xticklabels(["80", "90", "95", "99"])
        ax.set_xticks([98], minor=True)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("empirical coverage (%)")
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "..", "paper", "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "..", "paper", "figures", "calibration.pdf"),
                metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print("wrote figures/calibration.pdf")


def sci(x):
    """LaTeX scientific notation usable in text and math mode, e.g. 3\\times10^{-14}."""
    if x == 0:
        return "0"
    m, e = f"{x:.0e}".split("e")
    return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"


def diagnostic_tables():
    """Section 3.3-3.4 diagnostics: widths and HC3/HC1 inflation along the skewness sweep, the
    scale-versus-shape values for s = 2, the transformation check, the plug-in bias check and
    leverage-one Monte Carlo standard errors."""
    S = load("correction_summary.json")["designs"]
    vals, rows = {}, []
    for s_ in [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
        d = S[f"lognormal_s{s_:.1f}_prop"]; w = d["median_width_95"]; r = d["se_ratio_HC3_HC1"]
        rows.append(f"{s_:.1f} & " + " & ".join(f"{w[m]:.2f}" for m in ("IREG_HC1", "IREG_HC3", "REG_HC1", "REG_HC3"))
                    + f" & {r['IREG']['median']:.2f} & {r['IREG']['p90']:.2f} & {r['REG']['median']:.2f} & {r['REG']['p90']:.2f} \\\\")
        for m in ("IREG_HC1", "IREG_HC3", "REG_HC1", "REG_HC3"):
            vals[f"diag:s{s_:.1f}:width:{m}"] = f"{w[m]:.2f}"
        for e in ("IREG", "REG"):
            for q in ("median", "p90", "p95"):
                vals[f"diag:s{s_:.1f}:seratio:{e}:{q}"] = f"{r[e][q]:.2f}"
    write("gamma0_width_body.tex", rows)
    rows = []
    for s_ in [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
        sm = S[f"lognormal_s{s_:.1f}_prop"]["sampling"]; I, Rg = sm["IREG"], sm["REG"]
        from run_example import population
        pred = 1 + population(s_, 1.0, 0.0)["Delta"] / S[f"lognormal_s{s_:.1f}_prop"]["n"]
        predtxt = f"{pred:.2f}" if pred < 10 else f"{pred:.0f}"
        vals[f"samp:s{s_:.1f}:pred"] = predtxt
        sdth = (population(s_, 1.0, 0.0)["V"] / S[f"lognormal_s{s_:.1f}_prop"]["n"]) ** 0.5
        vals[f"samp:s{s_:.1f}:sdtheory"] = f"{sdth:.2f}"
        mr, mrse = sm["mse_ratio"], sm["mse_ratio_mcse"]
        vals[f"samp:s{s_:.1f}:mseratio:se"] = f"{mrse:.3f}"
        rows.append(f"{s_:.1f} & {I['sd']:.3f} & {Rg['sd']:.3f} & {sdth:.3f} & {mr:.3f}\\,({(f"{1e3 * mrse:.0f}" if round(1e3 * mrse) >= 1 else "$<$1")}) & {predtxt} & "
                    f"{I['median_se_HC1']:.3f} & {Rg['median_se_HC1']:.3f} & {I['median_se_HC3']:.3f} \\\\")
        vals[f"samp:s{s_:.1f}:sdI"], vals[f"samp:s{s_:.1f}:sdR"] = f"{I['sd']:.2f}", f"{Rg['sd']:.2f}"
        vals[f"samp:s{s_:.1f}:mseratio"] = f"{I['mse'] / Rg['mse']:.2f}"
        vals[f"samp:s{s_:.1f}:seI"], vals[f"samp:s{s_:.1f}:seR"] = f"{I['median_se_HC1']:.2f}", f"{Rg['median_se_HC1']:.2f}"
        vals[f"samp:s{s_:.1f}:sdpct"] = f"{100 * (I['sd'] / Rg['sd'] - 1):.0f}"
        vals[f"samp:s{s_:.1f}:sepct"] = f"{100 * (I['median_se_HC1'] / Rg['median_se_HC1'] - 1):.0f}"
    write("sampling_body.tex", rows)
    # tail sensitivity of the deficiency (Appendix C): truncated lognormal laws, closed-form moments
    from run_example import population
    exj = load("example.json")
    g0s = S["lognormal_s2.0_prop"]["sampling"]
    tails = [("Heavy-tailed example ($s=1.8$)", 1.8, 1.5, 0.3, exj["heavy_tail_case"]["500"]["sim"], "heavy", exj["heavy_tail_case"]["500"]["se"]),
             ("Table \\ref{tab:validity}, $s=1.5$", 1.5, 1.0, 0.4, exj["sweep"]["1.5"]["500"]["sim"], "t9", exj["sweep"]["1.5"]["500"]["se"]),
             ("Section \\ref{sec:gamma0}, $\\gamma=0$, $s=2$", 2.0, 1.0, 0.0, g0s["mse_ratio"], "g0", g0s["mse_ratio_mcse"])]
    vals["mc:heavy:se"] = f"{exj['heavy_tail_case']['500']['se']:.3f}"
    rows = []
    fmt = lambda x: f"{x:.2f}" if x < 10 else f"{x:.1f}"
    for lab, sx, c, g, sim, key, sse in tails:
        cells = [1 + population(sx, c, g, q)["Delta"] / 500 for q in (None, 1 - 1e-4, 1 - 1e-6, 1 - 1e-8)]
        rows.append(f"{lab} & {sim:.2f}\\,({1e2 * sse:.0f}) & " + " & ".join(fmt(x) for x in cells) + " \\\\")
        vals[f"tail:{key}:sim"] = f"{sim:.2f}"
        for q, x in zip(("full", "4", "6", "8"), cells):
            vals[f"tail:{key}:{q}"] = fmt(x)
    write("tail_body.tex", rows)
    exi = load("example.json")
    ip, ir, ic = exi["intro_population"], exi["intro_ratio"], exi["intro_coverage"]
    pc2 = lambda x: f"{x:.1f}" if abs(x) >= 2 else f"{x:.2f}"
    vals["intro:skew"], vals["intro:Delta"], vals["intro:Deltaround"] = f"{ip['skew']:.1f}", f"{ip['Delta']:.1f}", f"{ip['Delta']:.0f}"
    for n in ("500", "2000", "8000"):
        vals[f"intro:sim:{n}"], vals[f"intro:pred:{n}"] = pc2(100 * (ir[n]["sim"] - 1)), pc2(100 * (ir[n]["predicted"] - 1))
    vals["intro:se:500"] = f"{100 * ir['500']['se']:.2f}"
    for k in ("IREG+HC1", "IREG+HC3", "REG+HC1"):
        vals["intro:cov:" + k.replace("+", "_")] = f"{100 * ic[k][0]:.1f}"
    vals["intro:cov:mcse"] = f"{100 * max(ic[k][1] for k in ('IREG+HC1', 'IREG+HC3', 'REG+HC1')):.2f}"
    ex = load("example.json")["sweep"]
    for s_ in ("0.8", "1.0", "1.2"):
        vals[f"def:s{s_}"] = f"{ex[s_]['Delta']:.0f}"
        for n in ("500", "2000", "8000"):
            vals[f"defsim:s{s_}:n{n}"] = f"{100 * (ex[s_][n]['sim'] - 1):.1f}"
            vals[f"defpred:s{s_}:n{n}"] = f"{100 * (ex[s_][n]['predicted'] - 1):.1f}"
    for s_ in ("0.6", "0.8"):
        for n in ("500", "2000"):
            c = ex[s_][n]
            vals[f"valid:s{s_}:n{n}:frac"] = f"{100 * (c['sim'] - 1) / (c['predicted'] - 1):.0f}"
    # scale-versus-shape diagnostic for s = 2: the values quoted in Appendix F (no table body; the paper reports them in prose)
    st = S["lognormal_s2.0_prop"]["studentized"]
    lv = ["0.80", "0.90", "0.95", "0.98", "0.99"]
    for lab, name in (("IREG_HC1", "IREG, HC1"), ("IREG_HC3", "IREG, HC3"), ("REG_HC1", "REG, HC1")):
        x = st[lab]; c = x["implied_scale_from_95"]
        vals[f"diag:scale:{lab}"] = f"{c:.2f}"
        for l in lv:
            vals[f"diag:scale:{lab}:{l}:obs"] = f"{100 * x['levels'][l]['observed']:.1f}"
            vals[f"diag:scale:{lab}:{l}:pred"] = f"{100 * x['levels'][l]['scale_predicted']:.1f}"
            vals[f"diag:scale:{lab}:{l}:qratio"] = f"{x['levels'][l]['quantile_ratio']:.2f}"
        vals[f"diag:scale:{lab}:maxdev"] = f"{100 * max(abs(x['levels'][l]['observed'] - x['levels'][l]['scale_predicted']) for l in lv):.1f}"
    # transformation check (same s = 2 datasets)
    T = load("r_transform_check.json")
    for cv in ("X", "log1pX"):
        vals[f"tr:{cv}:lev"] = f"{T[cv]['median_max_leverage_IREG']:.2f}"
        for m in ("IREG_HC1", "IREG_HC3", "REG_HC1", "REG_HC3"):
            vals[f"tr:{cv}:{m}:cov"] = f"{100 * T[cv][m]['coverage']:.1f}"
            vals[f"tr:{cv}:{m}:width"] = f"{T[cv][m]['median_width']:.2f}"
    # plug-in bias: E[plug] ~ G/n + 4 p sigma^2 / n^2 in the Gaussian homoskedastic designs (sigma^2 = 1)
    gd = ["gauss_iid_n500_p5", "gauss_indep_arms_n500_p5", "gauss_iid_n1000_p2", "gauss_iid_n1000_p5", "gauss_iid_n1000_p10"]
    err = [abs(S[k]["plugin"]["mean"] / (S[k]["G"] / S[k]["n"] + 4 * S[k]["p"] / S[k]["n"] ** 2) - 1) for k in gd]
    vals["bias:gauss:maxrelerr"] = f"{100 * max(err):.1f}"
    vals["bias:gauss:ndesigns"] = str(len(gd))
    # REG coverage for the population ATE in the large-heterogeneity design
    L = S["large_hte_n500_p5"]["calibration"]
    vals["corr:large_hte_n500_p5:REG_HC1:0.95"] = f"{100 * L['0.95']['REG_HC1']['coverage']:.1f}"
    # Monte Carlo standard errors of the leverage-one subset coverages (Table 15)
    mc = []
    for f in ("r_gamma0_rare.json", "r_gamma0_rare_estimatr2.json"):
        for v in load(f).values():
            for e in ("IREG_HC1", "IREG_HC2", "IREG_HC3", "REG_HC1", "REG_HC3"):
                x = v[e]["leverage_one"]
                if x["mcse"] is not None:
                    mc.append(x["mcse"])
            vals.setdefault("lev1:n:min", str(min(v["IREG_HC1"]["leverage_one"]["n"] for v in load(f).values())))
    vals["lev1:mcse:max"] = f"{100 * max(mc):.1f}"
    rho, m = 0.02, 250                                   # rare-value design: X1 rare with prob rho, arms of n/2
    q0, q1 = (1 - rho) ** m, m * rho * (1 - rho) ** (m - 1)
    vals["rare:theory:rankdef"] = f"{100 * (1 - (1 - q0) ** 2):.2f}"
    vals["rare:theory:lev1"] = f"{100 * q1 * (2 * (1 - q0) - q1):.2f}"
    va = load("validation.json")["cexp2_full"]
    vals["val:full:dbern"] = f"{va['formula'] - 4 * va['G']:.2f}"
    e200, e800, e3200 = (va["est"][n]["cv"] for n in ("200", "800", "3200"))
    inc1, inc2 = e800 - e200, e3200 - e800
    ratio = inc2 / inc1
    vals["val:full:inc1"], vals["val:full:inc2"] = f"{inc1:.1f}", f"{inc2:.1f}"
    vals["val:full:extrap"] = f"{e3200 + inc2 * ratio / (1 - ratio):.0f}"
    return vals


if __name__ == "__main__":
    validation(); ablation(); validity(); gamma0_tables()
