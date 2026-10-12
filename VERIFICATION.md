# Verification

What in the paper is proved, what is verified by computation, and how to check it. Numbering follows
the paper (v1.1.0).

## Theory

- **Theorem 2** (second-order MSE difference, remainder `O(n^-3)`) holds under Assumption 1 alone. Its proof
  (Appendix A) uses the two-sample representation of the completely randomized design (Lemma 6), an exact
  expansion of the conditional expansion MSE at any treatment fraction (Lemma 7), the comparison between the
  Bernoulli and conditional expansions (Lemma 8), the CRT-to-Bernoulli reduction (Theorem 9) and the Bernoulli
  calculation (Theorem 10); Lemma 11 rewrites the result in matrix form.
- The index algebra of Theorem 10 is certified symbolically by `verification/general_p_symbolic_certificate.py`
  (zero symbolic remainder for general `p`). Before the main computation it asserts that its hand-coded first
  derivatives of the coefficient maps equal the generic formula `D theta[h] = Omega (db - dA theta)` with
  `theta_R = (0, gamma/2, 0)` and `theta_I = (0, 0, 0, gamma)` (57 components). The CRT shift
  `D_CRT = D_Bern + 4G` is checked numerically by `verification/crt_correction_certificate.py`. These are
  computer-assisted checks, not independent proofs.
- Remark 3 states the leading term of the expansion bias with an `O(n^-2)` remainder; the two leading terms are
  derived in Appendix A (Lemma 13) and compared with simulated biases in Appendix B. The expansion MSE is not the
  exact MSE; conditions relating them are not verified.
- In the heaviest-tailed designs the full-moment deficiency is dominated by covariate values beyond the
  `1 - 10^-4` quantile; Table 7 recomputes it for truncated lognormal laws (closed-form moments).

## Numerical verification of the theory

`simulations/run_validation.py` estimates `D_p` in six analytic designs with an exactly centered control
variate (Table 3); `run_ablation.py` shows that the verification rejects each incorrect variant of the formula
(Table 4); `run_validation_families.py` covers four covariate families at `p` in {2, 5, 8} (Table 5, Figures 1
and 2); `run_example.py` compares predicted and simulated MSE ratios along a lognormal family (Table 6, with
cell-specific Monte Carlo standard errors).

## Inference simulations (Section 3)

- Engine: `simulations/correction_python.py`, a batched NumPy implementation of REG and IREG (as fitted by
  `estimatr::lm_robust` and `estimatr::lm_lin`) with HC1/HC3 standard errors, the plug-in
  `delta' Sigma_X delta / n`, and the delete-one jackknife of eq. (12). The jackknife draws no random numbers,
  so adding it left every previously stored output of `results/hte_battery.json` unchanged (checked value by
  value when it was added).
- The jackknife uses the exact deletion formula for least squares; for IREG the leave-one-out fit is recentred
  at the leave-one-out covariate mean. A leave-one-out fit whose design is rank deficient (leverage one) is
  omitted, and eq. (12) is applied to the remaining estimates. `simulations/check_jackknife.py` compares the
  batched values with brute-force refitting on the remaining `n - 1` units, including datasets with omitted
  fits (relative gap about `10^-14`, `results/jackknife_check.json`).
- `r/validate_estimatr.R` confirms that the R engine reproduces the `lm_robust()`/`lm_lin()` formula interfaces,
  and `simulations/crosscheck_r.py` confirms agreement of estimates, standard errors, degrees of freedom,
  confidence limits and leverages between estimatr and the NumPy implementation (`simulations/hc.py`, the same
  fitting code as the engine) and statsmodels.
- All 14 designs of Tables 1–2 (defined in Appendix D) use 20,000 replications at `n = 500` and fixed seeds in
  `simulations/hte_designs.py`.

## Consistency and determinism

- Every number in Section 3, the Discussion and most of the appendices is generated from `results/` into
  `paper/tables/gamma0_values.tex` and the table bodies. `simulations/check_prose_numbers.py` recomputes every
  other number quoted in the prose, and the qualitative statements about the simulations, from `results/`, and
  checks that the quoted text occurs in `paper/main.tex`.
- `simulations/check_determinism.py` (Python) and a repeated run of `r/gamma0_sim.R` (R) confirm
  byte-identical output across fresh processes. All seeds are fixed integers.
- `run_all.sh` stops on any LaTeX warning; `make_arxiv_bundle.sh` test-compiles the upload bundle with pdflatex
  alone.

The plug-in diagnostic means in `hte_battery.json` exclude rejected IREG fits, whose interaction
coefficients are not identified. This changes only the ancillary plug-in summaries for the rare-value
design; the reported coverage, widths and MSE summaries are unchanged. In that design, MSE uses the
replications with both point estimates available, coverage counts unavailable intervals as failures,
and median width uses available intervals. The manuscript also reports the sensitivity to declaring
the jackknife interval unavailable whenever any deletion is undefined.

The current source has not yet had a verified clean compile after the targeted revisions; a Windows
access-denied error prevented the editor compiler from starting. The stale compiled PDF was removed.

## Not verified

Cross-platform reproducibility (tested on Linux x86_64 only); the sharpness of the moment condition; the
relation between the expansion MSE and trimmed exact moments.
