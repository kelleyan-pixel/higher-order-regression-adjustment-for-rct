#!/usr/bin/env bash
# Reproduces every number, table and the PDF from a clean checkout.
# Run from anywhere; all paths are resolved relative to this file.
# Single-threaded; expect roughly 3 hours in total (about two thirds in R).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

python3 simulations/env_report.py
Rscript -e 'cat(R.version.string, "| estimatr", as.character(packageVersion("estimatr")), "\n")'

echo "=== symbolic certificate for the Bernoulli constant D_Bern ==="
python3 -u verification/general_p_symbolic_certificate.py

echo "=== numerical certificate for the Bernoulli-to-CRT reduction ==="
python3 -u verification/crt_correction_certificate.py

echo "=== verification of D_p ==="
python3 -u simulations/run_validation.py

echo "=== ablation ==="
python3 -u simulations/run_ablation.py 3200

echo "=== bias decomposition ==="
python3 -u simulations/run_bias.py 1000000

echo "=== introductory example and range of validity ==="
python3 -u simulations/run_example.py

echo "=== R: estimatr 2.0.0 (pinned GitHub commit) into a repository-local library ==="
bash r/install_estimatr2.sh
LIB2="$ROOT/.Rlib/estimatr2"

echo "=== R: validation of the inference engine (estimatr system version and 2.0.0) ==="
Rscript r/validate_estimatr.R
Rscript r/validate_estimatr.R --lib="$LIB2" --tag=_estimatr2
python3 -u simulations/crosscheck_r.py

echo "=== R: gamma = 0 inference simulations ==="
for g in skew misspec nsweep p5 rare; do Rscript r/gamma0_sim.R "$g"; done
Rscript r/gamma0_sim.R rare --lib="$LIB2" --tag=_estimatr2
Rscript r/transform_check.R                         # log(1+X) check on the s = 2 datasets

echo "=== diagnostic: standard vs superpopulation-corrected IREG intervals (same datasets) ==="
Rscript r/correction_sim.R
python3 -u simulations/correction_python.py example
python3 -u simulations/correction_python.py prev
python3 -u simulations/hte_designs.py
python3 simulations/correction_summarize.py

echo "=== determinism checks (Python and R) ==="
python3 -u simulations/check_determinism.py
Rscript r/gamma0_sim.R smoke --tag=_det1 && Rscript r/gamma0_sim.R smoke --tag=_det2
cmp results/r_gamma0_smoke_det1.json results/r_gamma0_smoke_det2.json && echo "R: identical output from two runs"
rm -f results/r_gamma0_smoke_det1.json results/r_gamma0_smoke_det2.json

echo "=== implementation cross-check (NumPy vs statsmodels) ==="
python3 -u simulations/crosscheck_statsmodels.py

echo "=== LaTeX tables and generated values ==="
python3 simulations/make_tables.py
python3 simulations/check_prose_numbers.py

echo "=== compile the paper ==="
# Four passes: the generated table bodies move a few labels between the
# second and third pass, so a fourth pass is needed to settle cross-references.
cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null
if grep -q "LaTeX Warning" main.log; then
  echo "LaTeX warnings remain:"; grep "LaTeX Warning" main.log; exit 1
fi
echo "done: paper/main.pdf"
cd .. && bash make_arxiv_bundle.sh
