# Verification

What in the paper is proved, what is verified by computation, and how to check it.

## Theory

- **Theorem 2** (second-order MSE difference, remainder `O(n^-3)`) holds under Assumption 1 alone. Its proof (Appendix A) uses the
  two-sample representation of the completely randomized design (Lemma 7), an exact expansion of the
  conditional expansion MSE at any treatment fraction (Lemma 8), the comparison between the Bernoulli and
  conditional expansions (Lemma 9), the CRT-to-Bernoulli reduction (Theorem 10) and the Bernoulli calculation
  (Theorem 11).
- The index algebra of Theorem 11 is certified symbolically by `verification/general_p_symbolic_certificate.py`
  (zero symbolic remainder for general `p`). Before the main computation it asserts that its hand-coded first
  derivatives of the coefficient maps equal the generic formula `D theta[h] = Omega (db - dA theta)` with
  `theta_R = (0, gamma/2, 0)` and `theta_I = (0, 0, 0, gamma)` (57 components). The CRT shift `D_CRT = D_Bern + 4G` is checked numerically by
  `verification/crt_correction_certificate.py`. These are computer-assisted checks, not independent proofs.
- Remark 3 states the leading term of the expansion bias with an `O(n^-2)` remainder; the two leading terms are derived
  in Appendix A (Lemma "Leading expansion bias") and compared with simulated biases in Appendix B. The expansion MSE is not the exact
  MSE; conditions relating them are not verified.
- In the heaviest-tailed designs the full-moment deficiency is dominated by covariate values beyond the
  `1 - 10^-4` quantile; Table 9 recomputes it for truncated lognormal laws (closed-form moments).

## Numerical verification of the theory

`simulations/run_validation.py` estimates `D_p` in six analytic designs with an exactly centered control
variate; `run_ablation.py` shows that the verification rejects each incorrect variant of the formula;
`run_example.py` compares predicted and simulated MSE ratios along a lognormal family (Table 8, with
cell-specific Monte Carlo standard errors).

## Inference simulations

- R engine (`r/`): calls estimatr's fitting routine with design matrices built as `lm_robust()` and `lm_lin()`
  build them. `r/validate_estimatr.R` confirms identical output to the formula interfaces on 100 datasets and
  agreement to about 10^-14 with a base-R matrix implementation; `simulations/crosscheck_r.py` confirms agreement
  of estimates, standard errors, degrees of freedom, confidence limits and leverages with NumPy and statsmodels.
- The leading-order plug-in bias `G/n + 4 p sigma^2 / n^2` (Section 3.2) is a heuristic; it matches the
  simulated mean plug-in in the five Gaussian homoskedastic designs to within 0.8%.
- Leverage-one behavior differs between estimatr 1.0.2 and the GitHub development version 2.0.0; both are
  applied to the same simulated datasets (same seeds) and reported (Table 15, Appendix F). The proportions of
  rank-deficient and leverage-one replications match their exact probabilities within Monte Carlo error.

## Consistency and determinism

- Every number in Section 3 and the appendices, and the values in all tables, are generated from `results/`.
  `simulations/check_prose_numbers.py` recomputes every other number quoted in the prose, and the qualitative
  claims about the diagnostics, from `results/`.
- `simulations/check_determinism.py` (Python) and a repeated run of `r/gamma0_sim.R` (R) confirm
  byte-identical output across fresh processes. All seeds are fixed integers.
- `run_all.sh` stops on any LaTeX warning; `make_arxiv_bundle.sh` test-compiles the upload bundle with pdflatex
  alone.

## Not verified

Cross-platform reproducibility (tested on Linux x86_64 only); the sharpness of the moment condition; the
relation between the expansion MSE and trimmed exact moments.
