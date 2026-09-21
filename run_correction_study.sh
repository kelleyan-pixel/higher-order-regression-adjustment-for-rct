#!/usr/bin/env bash
# Diagnostic study (not part of the paper): standard vs superpopulation-corrected IREG intervals,
# on exactly the datasets behind every reported uncorrected coverage.  About one hour single-threaded.
# The raw per-replication R output (results/correction/r_*.json, ~9 MB per design) is regenerated
# here and not tracked; design-level summaries and the report are.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
Rscript r/correction_sim.R                          # 18 gamma = 0 designs of the paper (resumable)
python -u simulations/correction_python.py example   # introductory example + heavy-tailed variant
python -u simulations/correction_python.py prev      # previous-release designs (incl. large HTE)
python simulations/correction_summarize.py          # writes results/correction_summary.{json,csv,md}
echo "notes: docs/correction_diagnostic.md; results: results/correction_summary.md, results/calibration.csv"
