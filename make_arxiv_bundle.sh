#!/usr/bin/env bash
# Builds the arXiv upload bundle (arxiv_upload.tar.gz) from the compiled paper and test-compiles it
# in an empty directory with pdflatex only, as arXiv does (arXiv does not run BibTeX, so main.bbl is
# included). Run after run_all.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
test -f paper/main.bbl || { echo "paper/main.bbl missing: compile the paper first"; exit 1; }
TMP="$(mktemp -d)"
mkdir -p "$TMP/bundle/tables" "$TMP/bundle/figures"
cp paper/main.tex paper/main.bbl paper/references.bib "$TMP/bundle/"
cp paper/tables/*.tex "$TMP/bundle/tables/"
cp paper/figures/*.pdf "$TMP/bundle/figures/"
tar czf arxiv_upload.tar.gz -C "$TMP/bundle" .
mkdir "$TMP/test" && tar xzf arxiv_upload.tar.gz -C "$TMP/test"
( cd "$TMP/test" && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null )
if grep -q "LaTeX Warning\|Overfull\|undefined" "$TMP/test/main.log"; then
  echo "bundle compiles with warnings:"; grep "LaTeX Warning\|Overfull\|undefined" "$TMP/test/main.log"; exit 1
fi
echo "arxiv_upload.tar.gz: $(tar tzf arxiv_upload.tar.gz | wc -l) files; compiles cleanly with pdflatex alone"
rm -rf "$TMP"
