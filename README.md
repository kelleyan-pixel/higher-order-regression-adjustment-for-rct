# Higher-Order Asymptotics of Regression-Adjusted Estimators in Randomized Trials

Replication package for Anna Kelley and Stefan Wager, *Higher-Order Asymptotics of Regression-Adjusted
Estimators in Randomized Trials*. The paper is `paper/main.pdf` (source `paper/main.tex`). This release
(v1.1.0) matches the arXiv version: Sections 1–4 and Appendices A–D.

## Contents of the paper

- **Theory (main contribution).** In balanced completely randomized trials, simple regression adjustment
  (REG) and interacted regression adjustment (IREG, Lin 2013) have the same first-order variance. Theorem 2
  gives the closed-form difference in the `1/n^2` terms of their expansion MSEs under Assumption 1 (IID
  sampling and moments). The proof (Appendix A) uses the two-sample representation of the completely
  randomized design (Lemmas 6–8), the CRT-to-Bernoulli reduction (Theorem 9) and the Bernoulli calculation
  (Theorem 10), whose index algebra is certified symbolically (`verification/`). Appendix B checks the
  constant by simulation; Appendix C studies when `1 + Delta/n` is accurate.
- **Inference (simulation, Section 3).** IREG with HC1 plus the superpopulation plug-in, IREG with the
  delete-one jackknife, REG with HC1 and REG with the jackknife, across 14 designs (Tables 1–2, defined in
  Appendix D). The jackknife omits leave-one-out fits that are rank deficient (a unit with leverage one);
  the rare-value design is the only one where this occurs in more than 0.1% of replications.

## Layout

```
paper/            main.tex, references.bib, main.pdf; tables/ and figures/ are generated
verification/     symbolic certificate for Theorem 10; numerical check of Theorem 9
simulations/      Python: D_p verification, examples, inference simulations, cross-checks, table builder
  formula.py dgps.py sim.py            exact D_p, analytic designs, control-variate engine
  run_validation.py run_ablation.py run_bias.py run_example.py run_validation_families.py
  correction_python.py hte_designs.py  inference engine (HC1, HC3, plug-in, jackknife) and Section 3 designs
  check_jackknife.py                   jackknife deletion formula against brute-force refitting
  correction_common.py correction_summarize.py
  hc.py crosscheck_r.py crosscheck_statsmodels.py check_determinism.py
  check_prose_numbers.py make_tables.py env_report.py
r/                R: estimatr-based simulations and validation of the NumPy engine
reference/supplementary_designs/   coverages reported for the supplementary designs (reproduction check input)
results/          stored outputs of all simulations (JSON/CSV)
docs/             notes on the superpopulation-correction diagnostic (supplementary)
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
  `sudo apt-get install r-base-core r-cran-estimatr r-cran-jsonlite`). R is used for the `gamma = 0`
  lognormal designs quoted in Appendix C and for validating the NumPy engine against `estimatr`; estimatr's
  default without clustering is HC2, so HC1 and HC3 are requested explicitly.
- The supplementary rare-value runs also use the development version of estimatr on GitHub (DESCRIPTION
  version 2.0.0, commit `e70f3ee4c63eb5ea3803215cf5ac155c4feecced`), which handles leverage-one observations
  differently. `r/install_estimatr2.sh` builds it into `.Rlib/estimatr2`.
- LaTeX: pdflatex and bibtex with standard packages.

## Reproducing

From the repository root (single-threaded, about three hours):

```bash
bash run_all.sh
```

This runs the certificates, the verification of `D_p`, the examples, the R validation and cross-checks,
the inference simulations, the jackknife check, the determinism checks, table and figure generation, a
check of every number and qualitative statement quoted in the prose, the compilation of `paper/main.pdf`
(which fails on any LaTeX warning) and `make_arxiv_bundle.sh`. All seeds are fixed integers.

To rebuild the paper from the stored results only (a few minutes):

```bash
python3 simulations/correction_summarize.py
python3 simulations/make_tables.py
python3 simulations/check_prose_numbers.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && pdflatex main && cd ..
bash make_arxiv_bundle.sh
```

To rerun only the Section 3 simulations (about 10 minutes): `python3 -u simulations/hte_designs.py battery`,
then `python3 simulations/make_tables.py`.

## Stored versus generated

`results/`, `paper/tables/` and `paper/figures/` are generated and stored. The raw per-replication output
of `r/correction_sim.R` (about 9 MB per design) is not stored; `correction_summarize.py` caches a small
summary per design (`results/correction/rsum_*.json`), which is sufficient to regenerate every table.
`make_tables.py` also writes tables from earlier versions of the paper (`gamma0_*`, `hte_*`, `calibration_*`,
`correction95_*`, `null_main_*`, `sampling_*`); the current paper does not include them.

## Which script produces what

| Paper object | Script | Output |
|---|---|---|
| Theorem 10 (index algebra of the Bernoulli constant) | `verification/general_p_symbolic_certificate.py` | prints the certificate; zero remainder |
| Theorem 9 (CRT shift `D_CRT = D_Bern + 4G`) check | `verification/crt_correction_certificate.py` | `results/crt_correction.json` |
| Introductory example; Table 6; Appendix C heavy-tailed example | `simulations/run_example.py` | `results/example.json` |
| Tables 1 and 2; Section 3 and Discussion prose values | `simulations/hte_designs.py battery` (engine `correction_python.py`) | `results/hte_battery.json` |
| Jackknife implementation (Section 3.1) | `simulations/check_jackknife.py` | `results/jackknife_check.json` |
| NumPy engine against estimatr (Section 3.1) | `r/validate_estimatr.R`, `simulations/crosscheck_r.py`, `crosscheck_statsmodels.py` | `results/r_validation*.json`, `results/r_crosscheck.json` |
| Tables 3 and 4; Remark 3 bias check | `run_validation.py`, `run_ablation.py`, `run_bias.py` | `results/validation.json`, `ablation.json`, `bias.json` |
| Table 5, Figures 1 and 2 | `simulations/run_validation_families.py` | `results/validation_families.json` |
| Table 7 (tail sensitivity) | `simulations/make_tables.py` (closed-form truncated lognormal moments via `run_example.population`) | `paper/tables/tail_body.tex` |
| Appendix C, `gamma = 0` standard deviations; Table 7 last row | `r/correction_sim.R`, `simulations/correction_summarize.py` | `results/correction_summary.*` |
| All generated numbers in the prose | `simulations/make_tables.py` | `paper/tables/gamma0_values.tex` |

## Limitations

The inference results are simulation evidence. Leverage-one behavior depends on how software evaluates a
`0/0`; the jackknife omits undefined leave-one-out estimates, and the paper reports how coverage changes
under the alternative of counting those replications as failures. Reproducibility was verified on Linux
x86_64 with the versions listed.
