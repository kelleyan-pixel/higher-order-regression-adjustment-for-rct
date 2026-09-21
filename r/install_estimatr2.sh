#!/usr/bin/env bash
# Install estimatr 2.0.0 (GitHub, pinned commit) into a repository-local R library,
# alongside whatever estimatr version is installed system-wide (1.0.2 in the
# reference environment).  Used only for the rare-value comparison of Table 5.
# Requires: R with Rcpp, RcppEigen, Formula, generics, rlang; a C++ toolchain; network
# access to codeload.github.com.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA=e70f3ee4c63eb5ea3803215cf5ac155c4feecced
LIB="$ROOT/.Rlib/estimatr2"
if Rscript -e ".libPaths(c('$LIB', .libPaths())); stopifnot(packageVersion('estimatr', lib.loc='$LIB') == '2.0.0')" >/dev/null 2>&1; then
  echo "estimatr 2.0.0 already installed in $LIB"; exit 0
fi
mkdir -p "$LIB"
TMP="$(mktemp -d)"
curl -sSL -o "$TMP/estimatr.tar.gz" "https://codeload.github.com/DeclareDesign/estimatr/tar.gz/$SHA"
tar xzf "$TMP/estimatr.tar.gz" -C "$TMP"
R CMD INSTALL --no-docs --no-html --library="$LIB" "$TMP/estimatr-$SHA"
rm -rf "$TMP"
