"""Summarize the standard-vs-corrected IREG study and check reproduction.

Reads the raw per-replication R output (results/correction/r_<design>.json, written
by r/correction_sim.R; large, not tracked) and the Python design summaries
(results/correction/py_<design>.json), and writes

  results/correction_summary.json   full design-level results
  results/correction_summary.csv    one row per design (main coverage table)
  results/correction_summary.md     human-readable tables

Reproduction: for every design, the uncorrected coverages are compared with the
numbers already reported (results/r_gamma0_*.json and results/example.json for the
current paper; reference/supplementary_designs/ for the supplementary designs).

Usage: python simulations/correction_summarize.py
"""
import csv, glob, json, math, os
import numpy as np
from correction_common import summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
C = os.path.join(RES, "correction")
GROUPS = ["skew", "misspec", "nsweep", "p5", "rare"]


def r_summary(path):
    raw = json.load(open(path))
    d = raw["design"]; n, p = d["n"], d["p"]
    rows = raw["sims"]
    col = lambda k: np.array([r.get(k, math.nan) if r.get(k) is not None else math.nan for r in rows], float)
    from scipy import stats
    a = {"R_est": col("R_est1"), "R_se1": col("R_se1"), "R_se3": col("R_se3"),
         "I_est": col("I_est1"), "I_se1": col("I_se1"), "I_se2": col("I_se2"), "I_se3": col("I_se3"),
         "plug": col("plug")}
    for pre in ("R", "I"):
        df = col(f"{pre}_df1")
        a[f"{pre}_df"] = np.where(np.isfinite(df), df, math.nan)
        a[f"{pre}_q"] = np.where(np.isfinite(df), stats.t.ppf(0.975, np.where(np.isfinite(df), df, 1)), math.nan)
    # estimates / df are identical across HC fits; each HC's own est/df are used for its interval
    for k in ("2", "3"):
        assert np.allclose(col(f"I_est{k}"), a["I_est"], equal_nan=True)
    lev = col("levI"); rank = col("rankI")
    a["lev1"] = np.isfinite(lev) & (lev > 1 - 1e-8)
    a["rankdef"] = rank < 2 + 2 * p
    meta = dict(design=d["name"], source="current paper, Section 3 (r/gamma0_sim.R)",
                engine=f"R, estimatr {raw['estimatr_version']}", n=n, p=p, seed=raw["seed"], hte=False,
                truth=dict(gamma=[0.0] * p, Sigma_X="not needed: gamma = 0", G=0.0, V_star=None,
                           R2_tau=0.0, PATE=1.0, V_star_method="G = 0 exactly"))
    return summarize(a, 1.0, 0.0, n, meta)


def reported():
    rep = {}
    for g in GROUPS:
        for k, v in json.load(open(os.path.join(RES, f"r_gamma0_{g}.json"))).items():
            rep[k] = {e: v[e]["all"]["coverage"] for e in ["IREG_HC1", "IREG_HC2", "IREG_HC3", "REG_HC1", "REG_HC3"]}
    ex = json.load(open(os.path.join(RES, "example.json")))
    for key, cv in [("intro_example", ex["intro_coverage"]), ("heavy_tail_example", ex["heavy_tail_case"]["coverage"])]:
        rep[key] = {"IREG_HC1": cv["IREG+HC1"][0], "IREG_HC2": cv["IREG+HC2"][0], "IREG_HC3": cv["IREG+HC3"][0],
                    "REG_HC1": cv["REG+HC1"][0], "REG_HC3": cv["REG+HC3"][0], "IREG_HC1c": cv["IREG+HC1c"][0]}
    P = os.path.join(ROOT, "reference", "supplementary_designs")
    t1 = json.load(open(os.path.join(P, "table1.json"))); t3 = json.load(open(os.path.join(P, "table3.json")))
    for k, v in t1.items():
        rep[f"{k}_n500_p5"] = {"IREG_HC1": v["I_HC1_pate"][2], "IREG_HC3": v["I_HC3_pate"][2],
                               "REG_HC1": v["R_HC1_pate"][2], "REG_HC3": v["R_HC3_pate"][2],
                               "IREG_HC1c": v["I_HC1c_pate"][2], "IREG_HC3c": v["I_HC3c_pate"][2]}
    for k, v in t3.items():
        nm, p = k.rsplit("_p", 1)
        rep[f"{nm}_n1000_p{p}"] = {"IREG_HC1": v["I_HC1_pate"][2], "IREG_HC3": v["I_HC3_pate"][2],
                                   "REG_HC1": v["R_HC1_pate"][2], "REG_HC3": v["R_HC3_pate"][2],
                                   "IREG_HC3c": v["I_HC3c_pate"][2]}
    return rep


ORDER = (["intro_example", "heavy_tail_example"]
         + [f"{k}_n500_p5" for k in ["gauss_iid", "gauss_indep_arms", "skew_het", "heavy_het", "rare_extreme", "large_hte", "null_hte"]]
         + [f"{k}_n1000_p{p}" for k in ["gauss_iid", "skew_het", "large_hte", "misspec"] for p in (2, 5, 10)]
         + [f"lognormal_s{s:.1f}_prop" for s in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)]
         + ["lognormal_s1.2_prop_n200", "lognormal_s1.2_prop_n2000", "lognormal_s1.2_prop_n8000",
            "gauss_quad", "lognormal_s1.0_log", "lognormal_s1.0_quadstd", "lognormal_s1.0_log_prop",
            "lognormal_s1.0_prop_p5_n500", "lognormal_s1.0_prop_p5_n200", "rare_q0.02_prop", "rare_q0.02_homo"])


def write_md(sums):
    pc = lambda x: "--" if x is None else f"{100 * x:.1f}"
    L = ["# Standard vs superpopulation-corrected IREG intervals", "",
         "Coverage (%) of the PATE by nominal 95% intervals. `c` = plug-in correction "
         "(+ delta' Sigma_X delta / n); `o` = oracle correction (+ G/n, diagnostic only). "
         "Monte Carlo standard errors are in `correction_summary.csv`.", "",
         "## Coverage", "",
         "| Design | HTE | n | p | R2_tau | IREG HC1 | HC1c | HC2 | HC2c | HC3 | HC3c | REG HC1 | REG HC3 | IREG HC1o | HC3o | max MCSE |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in [k for k in ORDER if k in sums]:
        s = sums[k]; cv = s["coverage"]
        cells = [cv[c]["coverage"] for c in ["IREG_HC1", "IREG_HC1c", "IREG_HC2", "IREG_HC2c", "IREG_HC3",
                                              "IREG_HC3c", "REG_HC1", "REG_HC3"]]
        o = [cv.get("IREG_HC1o", {}).get("coverage"), cv.get("IREG_HC3o", {}).get("coverage")]
        mc = max(v["mcse"] for v in cv.values())
        L.append(f"| {k} | {'yes' if s['hte'] else 'no'} | {s['n']} | {s['p']} | {s['truth']['R2_tau']:.3f} | "
                 + " | ".join(pc(c) for c in cells) + " | " + " | ".join(pc(c) for c in o) + f" | {100 * mc:.2f} |")
    L += ["", "## Plug-in correction vs the true first-order component (IREG)", "",
          "Plug-in statistics use full-rank IREG fits; coverage above uses every replication.", "",
          "| Design | G/n (true) | mean plug-in | median plug-in | mean/(G/n) | median plug-in / HC1 var | "
          "HC1 median width ratio | HC1 miss->cover | HC3 miss->cover | rank-def. | degenerate fit | lev. one |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in [k for k in ORDER if k in sums]:
        s = sums[k]; pl = s["plugin"]; w = s["width"]; f = s["failures"]
        r = "--" if pl["ratio_mean_to_true"] is None else f"{pl['ratio_mean_to_true']:.2f}"
        L.append(f"| {k} | {s['G_over_n']:.2e} | {pl['mean']:.2e} | {pl['median']:.2e} | {r} | "
                 f"{pl['median_plug_over_HC1_var']:.3f} | {w['HC1']['median_width_ratio']:.3f} | "
                 f"{pc(w['HC1']['frac_miss_to_cover'])} | {pc(w['HC3']['frac_miss_to_cover'])} | "
                 f"{pc(f['frac_rank_deficient_IREG'])} | {pc(f['frac_degenerate_IREG_fit'])} | {pc(f['frac_leverage_one_IREG'])} |")
    open(os.path.join(RES, "correction_summary.md"), "w").write("\n".join(L) + "\n")


def main():
    sums = {}
    # R designs: summarize the raw output when present (and cache the small summary as
    # rsum_<design>.json); otherwise use the cached summary, so the published repository
    # can regenerate every table without the ~9 MB-per-design raw files.
    for f in sorted(glob.glob(os.path.join(C, "r_*.json"))):
        s = r_summary(f); sums[s["design"]] = s
        json.dump(s, open(os.path.join(C, f"rsum_{s['design']}.json"), "w"), indent=1)
    for f in sorted(glob.glob(os.path.join(C, "rsum_*.json"))):
        s = json.load(open(f)); sums.setdefault(s["design"], s)
    for f in sorted(glob.glob(os.path.join(C, "py_*.json"))):
        s = json.load(open(f)); sums[s["design"]] = s
    rep = reported()
    worst = 0.0
    for k, s in sums.items():
        chk = {e: (s["coverage"][e]["coverage"], rep[k][e]) for e in rep.get(k, {}) if e in s["coverage"]}
        s["reproduction"] = {e: {"here": a, "reported": b, "abs_diff": abs(a - b)} for e, (a, b) in chk.items()}
        if chk:
            worst = max(worst, max(abs(a - b) for a, b in chk.values()))
    out = {"designs": sums, "max_abs_diff_vs_reported": worst}
    json.dump(out, open(os.path.join(RES, "correction_summary.json"), "w"), indent=1)
    cols = ["IREG_HC1", "IREG_HC1c", "IREG_HC2", "IREG_HC2c", "IREG_HC3", "IREG_HC3c", "REG_HC1", "REG_HC3"]
    with open(os.path.join(RES, "correction_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "source", "hte", "n", "p", "reps", "G", "R2_tau", "PATE"] + cols
                   + [c + "_mcse" for c in cols] + ["IREG_HC1o", "IREG_HC3o", "plug_mean", "plug_median",
                   "G_over_n", "plug_mean_over_G_over_n", "HC1_width_ratio_median", "HC1_frac_miss_to_cover",
                   "frac_rank_deficient", "frac_leverage_one"])
        for k, s in sums.items():
            cv, t = s["coverage"], s["truth"]
            w.writerow([k, s["source"], s["hte"], s["n"], s["p"], s["reps"], t["G"], t["R2_tau"], t["PATE"]]
                       + [cv[c]["coverage"] for c in cols] + [cv[c]["mcse"] for c in cols]
                       + [cv.get("IREG_HC1o", {}).get("coverage"), cv.get("IREG_HC3o", {}).get("coverage"),
                          s["plugin"]["mean"], s["plugin"]["median"], s["G_over_n"], s["plugin"]["ratio_mean_to_true"],
                          s["width"]["HC1"]["median_width_ratio"], s["width"]["HC1"]["frac_miss_to_cover"],
                          s["failures"]["frac_rank_deficient_IREG"], s["failures"]["frac_leverage_one_IREG"]])
    write_md(sums)
    print(f"{len(sums)} designs summarized; largest |difference| from reported uncorrected coverage: {worst:.2e}")


if __name__ == "__main__":
    main()
