"""Shared summary statistics for the standard-vs-corrected IREG interval study.

Given per-replication arrays for one design, computes coverage of the PATE for
REG (HC1, HC3), IREG (HC1-HC3), IREG with the plug-in superpopulation correction
(HC1c-HC3c: Vhat + delta' Sigma_X delta / n) and, when G > 0, IREG with the *oracle*
correction (HC1o, HC3o: Vhat + G/n, G the true gamma' Sigma_X gamma).  The oracle
columns isolate what the first-order superpopulation component does by itself;
they are a diagnostic, not a feasible procedure.

A replication covers only if the interval is finite; undefined intervals count as
non-covers (procedure-level coverage), matching the paper.
"""
import numpy as np

LEVELS = (0.80, 0.90, 0.95, 0.98, 0.99)


def _cov(hit):
    c = float(np.mean(hit))
    return {"coverage": c, "mcse": float(np.sqrt(c * (1 - c) / hit.size))}


def summarize(a, tau, G, n, meta):
    """a: dict of arrays with keys
       R_est, R_se{1,3}, R_q, R_df, I_est, I_se{1,2,3}, I_q, I_df, plug, lev1 (bool), rankdef (bool).
       R_q, I_q are the 97.5% t quantiles at R_df, I_df (the paper's 95% intervals)."""
    out = dict(meta)
    B = a["I_est"].size
    out.update({"reps": int(B), "G": float(G), "G_over_n": float(G / n)})
    def hits(est, se, q):
        ok = np.isfinite(est) & np.isfinite(se) & np.isfinite(q)
        return ok & (np.abs(np.where(ok, est, 0) - tau) <= np.where(ok, q * se, -1))
    cov = {}
    for k in ("1", "3"):
        cov[f"REG_HC{k}"] = _cov(hits(a["R_est"], a[f"R_se{k}"], a["R_q"]))
    width = {}
    for k in ("1", "2", "3"):
        se = a[f"I_se{k}"]
        sec = np.sqrt(se ** 2 + a["plug"])
        h, hc = hits(a["I_est"], se, a["I_q"]), hits(a["I_est"], sec, a["I_q"])
        cov[f"IREG_HC{k}"], cov[f"IREG_HC{k}c"] = _cov(h), _cov(hc)
        if G > 0:
            cov[f"IREG_HC{k}o"] = _cov(hits(a["I_est"], np.sqrt(se ** 2 + G / n), a["I_q"]))
        ok = np.isfinite(se) & np.isfinite(sec)
        ratio = sec[ok] / se[ok]
        width[f"HC{k}"] = {
            "mean_width": float(np.mean(2 * a["I_q"][ok] * se[ok])),
            "mean_width_corrected": float(np.mean(2 * a["I_q"][ok] * sec[ok])),
            "median_width_ratio": float(np.median(ratio)),
            "frac_width_up_5pct": float(np.mean(ratio > 1.05)),
            "frac_coverage_flips": float(np.mean(h != hc)),
            "frac_miss_to_cover": float(np.mean(~h & hc)),
            "frac_no_interval": float(np.mean(~np.isfinite(se))),
            # where the lengthening falls: replications in which the uncorrected interval misses vs covers
            "median_width_ratio_uncorrected_misses": float(np.median(ratio[~h[ok]])) if np.any(~h[ok]) else None,
            "median_width_ratio_uncorrected_covers": float(np.median(ratio[h[ok]])) if np.any(h[ok]) else None,
        }
    # Plug-in statistics use full-rank IREG fits only: in a rank-deficient fit the interaction
    # coefficients are not identified, and estimatr 1.0.2 sometimes returns numerically degenerate
    # coefficients there (counted below).  Coverage above uses every replication.
    full = ~a["rankdef"] & np.isfinite(a["plug"])
    plug = a["plug"][full]
    out["coverage"] = cov
    out["width"] = width
    out["plugin"] = {
        "mean": float(np.mean(plug)), "median": float(np.median(plug)),
        "sd": float(np.std(plug)), "true_G_over_n": float(G / n),
        "excess_mean_minus_true": float(np.mean(plug) - G / n),
        "ratio_mean_to_true": (float(np.mean(plug) / (G / n)) if G > 0 else None),
        # typical size of the plug-in relative to the uncorrected HC1 variance
        "median_plug_over_HC1_var": float(np.median(a["plug"][full] / a["I_se1"][full] ** 2)),
        "n_reps_used": int(full.sum()),
    }
    # Calibration across nominal levels: same estimate, same variance, same degrees of freedom;
    # only the t critical value changes.  The oracle (+ G/n) is computed for every design; when
    # G = 0 it coincides with the uncorrected interval by construction.
    from scipy import stats
    cal = {}
    for level in LEVELS:
        qI = stats.t.ppf(1 - (1 - level) / 2, a["I_df"])
        row = {}
        for k in ("1", "3"):
            se = a[f"I_se{k}"]
            row[f"IREG_HC{k}"] = _cov(hits(a["I_est"], se, qI))
            row[f"IREG_HC{k}c"] = _cov(hits(a["I_est"], np.sqrt(se ** 2 + a["plug"]), qI))
            row[f"IREG_HC{k}o"] = _cov(hits(a["I_est"], np.sqrt(se ** 2 + G / n), qI))
        row["REG_HC1"] = _cov(hits(a["R_est"], a["R_se1"], stats.t.ppf(1 - (1 - level) / 2, a["R_df"])))
        cal[f"{level:.2f}"] = row
    out["calibration"] = cal
    # Scale versus shape of the studentized statistic T = (est - tau) / SE.  If T were a scaled t,
    # T ~ c * t_df, coverage at level L would be 2 F_t(q_L / c) - 1; c is read off the observed 95%
    # coverage and used to predict the other levels.  The ratio of the empirical |T| quantile to the
    # t quantile at each level is the direct check: constant across levels = pure scale error.
    shape = {}
    for lab, est, se, dfa in [("IREG_HC1", a["I_est"], a["I_se1"], a["I_df"]),
                              ("IREG_HC3", a["I_est"], a["I_se3"], a["I_df"]),
                              ("REG_HC1", a["R_est"], a["R_se1"], a["R_df"])]:
        ok = np.isfinite(est) & np.isfinite(se) & np.isfinite(dfa)
        absT = np.abs(est[ok] - tau) / se[ok]
        df0 = float(np.median(dfa[ok]))
        c95 = cal["0.95"][lab]["coverage"]
        scale = stats.t.ppf(0.975, df0) / stats.t.ppf((1 + c95) / 2, df0)
        shape[lab] = {"df": df0, "implied_scale_from_95": float(scale), "levels": {}}
        for level in LEVELS:
            qL = stats.t.ppf(1 - (1 - level) / 2, df0)
            shape[lab]["levels"][f"{level:.2f}"] = {
                "observed": cal[f"{level:.2f}"][lab]["coverage"],
                "scale_predicted": float(2 * stats.t.cdf(qL / scale, df0) - 1),
                "quantile_ratio": float(np.quantile(absT, level) / qL)}
    out["studentized"] = shape
    # How much HC3 inflates the estimated uncertainty of tau_hat relative to HC1, and interval widths.
    def pct(x, q):
        x = x[np.isfinite(x)]
        return float(np.quantile(x, q))
    out["se_ratio_HC3_HC1"] = {
        "IREG": {"median": pct(a["I_se3"] / a["I_se1"], 0.5), "p90": pct(a["I_se3"] / a["I_se1"], 0.9),
                 "p95": pct(a["I_se3"] / a["I_se1"], 0.95)},
        "REG": {"median": pct(a["R_se3"] / a["R_se1"], 0.5), "p90": pct(a["R_se3"] / a["R_se1"], 0.9),
                "p95": pct(a["R_se3"] / a["R_se1"], 0.95)}}
    # Sampling variability of the estimators themselves versus their estimated standard errors.
    def samp(est, se1, se3):
        e = est[np.isfinite(est)]
        return {"sd": float(np.std(e, ddof=1)), "mse": float(np.mean((e - tau) ** 2)),
                "median_se_HC1": pct(se1, 0.5), "median_se_HC3": pct(se3, 0.5)}
    out["sampling"] = {"IREG": samp(a["I_est"], a["I_se1"], a["I_se3"]),
                       "REG": samp(a["R_est"], a["R_se1"], a["R_se3"])}
    out["median_width_95"] = {
        "IREG_HC1": pct(2 * a["I_q"] * a["I_se1"], 0.5), "IREG_HC3": pct(2 * a["I_q"] * a["I_se3"], 0.5),
        "REG_HC1": pct(2 * a["R_q"] * a["R_se1"], 0.5), "REG_HC3": pct(2 * a["R_q"] * a["R_se3"], 0.5)}
    out["failures"] = {"frac_rank_deficient_IREG": float(np.mean(a["rankdef"])),
                       "frac_degenerate_IREG_fit": float(np.mean(~np.isfinite(a["I_est"])
                                                                 | (np.abs(np.nan_to_num(a["I_est"]) - tau) > 1e6))),
                       "frac_leverage_one_IREG": float(np.mean(a["lev1"]))}
    return out
