# Higher-Order Asymptotics of Regression-Adjusted Estimators in Randomized Trials

Replication package for Anna Kelley and Stefan Wager, *Higher-Order Asymptotics of Regression-Adjusted
Estimators in Randomized Trials*. The paper is `paper/main.pdf` (source `paper/main.tex`).

## Contents of the paper

- **Theory (main contribution).** In balanced completely randomized trials, simple regression adjustment
  (REG) and interacted regression adjustment (IREG, Lin 2013) have the same first-order variance. Theorem 2
  gives the closed-form difference in the `1/n^2` terms of their expansion MSEs under Assumption 1 (IID
  sampling and moments); no further high-level assumption is needed. The proof uses the two-sample representation of the completely randomized design
  (Appendix A) and a symbolic certificate for the index algebra (`verification/`).
- **Inference (simulation).** Without treatment-effect heterogeneity, IREG with HC1 standard errors can
  undercover substantially under skewed, heteroskedastic designs, mainly through understatement of the
  variance scale; HC3 largely corrects the scale but not extreme leverage (Section 3).
- **Superpopulation component.** The conventional robust variance of the centered IREG implementation omits a
  component `G/n` of the population-ATE variance; REG's does not. Appendix E compares the plug-in correction
  with the true `G/n` across nominal levels.

## Layout

```
paper/            main.tex, references.bib, main.pdf; tables/ and figures/ are generated
verification/     symbolic certificate for Theorem 12; numerical check of Theorem 11
simulations/      Python: D_p verification, examples, correction diagnostic, cross-checks, table builder
  formula.py dgps.py sim.py            exact D_p, analytic designs, control-variate engine
  run_validation.py run_ablation.py run_bias.py run_example.py
  correction_python.py correction_common.py correction_summarize.py
  hc.py crosscheck_r.py crosscheck_statsmodels.py check_determinism.py
  check_prose_numbers.py make_tables.py env_report.py
r/                R: inference simulations with estimatr
  common.R gamma0_sim.R correction_sim.R transform_check.R validate_estimatr.R install_estimatr2.sh
reference/supplementary_designs/   coverages reported for the supplementary designs (reproduction check input)
results/          stored outputs of all simulations (JSON/CSV)
docs/             notes on the correction diagnostic
run_all.sh        reproduces everything, compiles the paper and builds the arXiv bundle
run_correction_study.sh            reruns only the correction diagnostic
make_arxiv_bundle.sh               builds and test-compiles arxiv_upload.tar.gz
VERIFICATION.md   what is verified, and how
LICENSE, CITATION.cff   MIT license and citation metadata
MANIFEST.txt, SHA256SUMS.txt   file list and checksums of this tree (check with: sha256sum -c SHA256SUMS.txt)
```

## Software

- Python 3.12 with `requirements.txt`. On Ubuntu 24.04, install into a virtual environment:
  `sudo apt-get install python3-venv`, then
  `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`.
  All scripts are invoked as `python3`. Requirements (numpy 2.4.4, scipy 1.17.1, sympy 1.14.0, statsmodels 0.15.0,
  matplotlib 3.10.8).
- R 4.3.3 with `estimatr` 1.0.2 and `jsonlite` (on Ubuntu 24.04:
  `sudo apt-get install r-base-core r-cran-estimatr r-cran-jsonlite`). All main inference results use
  estimatr 1.0.2 with `se_type = "HC1"`, `"HC2"` or `"HC3"` requested explicitly; estimatr's default without
  clustering is HC2.
- Table 6 also uses the development version of estimatr on GitHub (DESCRIPTION version 2.0.0, commit
  `e70f3ee4c63eb5ea3803215cf5ac155c4feecced`), which handles leverage-one observations differently.
  `r/install_estimatr2.sh` builds it into `.Rlib/estimatr2` (needs a C++ toolchain, Rcpp, RcppEigen, Formula,
  generics, rlang and network access to codeload.github.com). We do not claim it corresponds to a CRAN release.
- The R engine calls `estimatr:::lm_robust_fit`, the fitting routine behind `lm_robust()` and `lm_lin()`;
  `r/validate_estimatr.R` checks that the formula interfaces give identical results.
- LaTeX: pdflatex and bibtex with standard packages.

## Reproducing

From the repository root (single-threaded, about three hours):

```bash
bash run_all.sh
```

This runs the certificates, the verification of `D_p`, the examples, the R validation and cross-checks,
the inference simulations, the correction diagnostic, the determinism checks, table and figure
generation, a check of every number quoted in the prose, the compilation of `paper/main.pdf` (which fails
on any LaTeX warning) and `make_arxiv_bundle.sh`. All seeds are fixed integers.

To rebuild the paper from the stored results only (a few minutes):

```bash
python3 simulations/correction_summarize.py
python3 simulations/make_tables.py
python3 simulations/check_prose_numbers.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && pdflatex main && cd ..
bash make_arxiv_bundle.sh
```

## Stored versus generated

`results/`, `paper/tables/` and `paper/figures/` are generated and stored. The raw per-replication output
of `r/correction_sim.R` (about 9 MB per design) is not stored; `correction_summarize.py` caches a small
summary per design (`results/correction/rsum_*.json`), which is sufficient to regenerate every table.

## Which script produces what

| Paper object | Script | Output |
|---|---|---|
| Theorem 12 (index algebra) | `verification/general_p_symbolic_certificate.py` | prints the certificate; zero remainder |
| Theorem 11 check | `verification/crt_correction_certificate.py` | `results/crt_correction.json` |
| Introductory example, Table 9 | `simulations/run_example.py` | `results/example.json` |
| Tables 1, 2, 6 | `r/gamma0_sim.R` | `results/r_gamma0_*.json` |
| Tables 3, 4, 5; Tables 11, 12 and Figure 1 | `r/correction_sim.R`, `simulations/correction_python.py`, `simulations/correction_summarize.py` | `results/correction_summary.*`, `results/calibration.csv` |
| Table 10 (tail sensitivity), prediction column of Table 4 | `simulations/make_tables.py` (closed-form truncated lognormal moments via `run_example.population`) | `paper/tables/tail_body.tex` |
| Section 3.3, log(1+X) check | `r/transform_check.R` | `results/r_transform_check.json` |
| Tables 7, 8; Remark 3 | `run_validation.py`, `run_ablation.py`, `run_bias.py` | `results/validation.json`, `ablation.json`, `bias.json` |
| Appendix D | `r/validate_estimatr.R`, `simulations/crosscheck_r.py` | `results/r_validation*.json`, `results/r_crosscheck.json` |

## Limitations

The inference results are simulation evidence. Leverage-one behavior depends on how software evaluates a
`0/0`; results are given for the two estimatr versions above. Reproducibility was verified on Linux x86_64
with the versions listed.
