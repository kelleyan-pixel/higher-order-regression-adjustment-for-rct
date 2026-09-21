#!/usr/bin/env Rscript
# Validation of the R simulation engine against estimatr and an independent
# base-R implementation, plus the special-case checks described in VERIFICATION.md.
#
# 1. Formula interface: lm_robust(y ~ w + x) and lm_lin(y ~ w, covariates = ~ x)
#    give the same estimate, HC1/HC2/HC3 standard error, df and CI as the
#    estimatr:::lm_robust_fit calls used by r/gamma0_sim.R.
# 2. Hand implementation: base-R matrix formulas for OLS, leverage and HC1-HC3
#    reproduce estimatr on non-degenerate datasets.
# 3. Export: datasets and estimatr output are written to results/r_datasets/ for
#    the independent Python/statsmodels cross-check (simulations/crosscheck_r.py).
# 4. gamma = 0: individual treatment effects are identically tau, so SATE = PATE,
#    while the in-sample plug-in delta' Sigma delta / n is nonetheless positive.
# 5. p = 1: estimatr agrees with the closed-form within-arm expressions.
# 6. Leverage one: a treated arm with a single rare-value unit gives h_i = 1 and
#    e_i = 0 up to rounding; the HC2/HC3 output of the installed estimatr is recorded.
#
# Usage: Rscript r/validate_estimatr.R [--lib=/path/to/other/library] [--tag=suffix]
# Output: results/r_validation<tag>.json (+ results/r_datasets/ when tag is empty)

args <- commandArgs(trailingOnly = FALSE)
script <- normalizePath(sub("^--file=", "", args[grep("^--file=", args)]))
ROOT <- dirname(dirname(script))
targs <- commandArgs(trailingOnly = TRUE)
libarg <- sub("^--lib=", "", targs[grep("^--lib=", targs)])
tag <- sub("^--tag=", "", targs[grep("^--tag=", targs)])
if (length(tag) == 0) tag <- ""
if (length(libarg) == 1) .libPaths(c(libarg, .libPaths()))
suppressMessages(library(estimatr))
suppressMessages(library(jsonlite))
source(file.path(ROOT, "r", "common.R"))
set.seed(4242, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
report <- list(estimatr_version = as.character(packageVersion("estimatr")),
               R_version = R.version.string)
relgap <- function(a, b) max(abs(a - b) / pmax(abs(b), 1))

hand_hc <- function(y, M, col = 2) {
  n <- nrow(M); k <- ncol(M)
  A <- solve(crossprod(M)); b <- A %*% crossprod(M, y)
  e <- drop(y - M %*% b); h <- rowSums((M %*% A) * M)
  meat <- function(w) A %*% crossprod(M * sqrt(w)) %*% A
  V <- list(HC1 = meat(e^2 * n / (n - k)), HC2 = meat(e^2 / (1 - h)), HC3 = meat(e^2 / (1 - h)^2))
  list(est = b[col], se = sapply(V, function(v) sqrt(v[col, col])), h = h, e = e, df = n - k)
}

# ------------------------------------------------ 1-3: equivalence and export
designs <- list(
  lognormal_p1 = list(name = "v1", x = "lognormal", s = 1.2, het = "prop", mis = "none", p = 1, n = 200),
  lognormal_p5 = list(name = "v2", x = "lognormal", s = 1.0, het = "prop", mis = "none", p = 5, n = 200),
  gauss_misspec = list(name = "v3", x = "gauss", s = NA, het = "none", mis = "quad_gauss", p = 1, n = 200),
  rare_het = list(name = "v4", x = "rare", q = 0.05, het = "prop", mis = "none", p = 1, n = 200))
gap_formula <- 0; gap_hand <- 0; n_checked <- 0
if (tag == "") dir.create(file.path(ROOT, "results", "r_datasets"), recursive = TRUE, showWarnings = FALSE)
for (dn in names(designs)) {
  d <- designs[[dn]]
  for (b in 1:25) {
    z <- draw(d)
    df <- data.frame(y = z$y, w = z$w, z$X); xn <- paste0("x", seq_len(d$p)); names(df)[-(1:2)] <- xn
    MR <- cbind(1, z$w, z$X); Xc <- sweep(z$X, 2, colMeans(z$X)); MI <- cbind(1, z$w, Xc, z$w * Xc)
    if (qr(MI)$rank < ncol(MI)) next
    fr <- as.formula(paste("y ~ w +", paste(xn, collapse = " + ")))
    fc <- as.formula(paste("~", paste(xn, collapse = " + ")))
    out <- list()
    for (se in c("HC1", "HC2", "HC3")) {
      a <- lm_robust(fr, data = df, se_type = se); l <- lm_lin(y ~ w, covariates = fc, data = df, se_type = se)
      eR <- fit_one(z$y, MR, se); eI <- fit_one(z$y, MI, se)
      gap_formula <- max(gap_formula,
        relgap(c(a$coefficients["w"], a$std.error["w"], a$df["w"]), eR),
        relgap(c(l$coefficients["w"], l$std.error["w"], l$df["w"]), eI))
      hR <- hand_hc(z$y, MR); hI <- hand_hc(z$y, MI)
      gap_hand <- max(gap_hand, relgap(c(hR$est, hR$se[se], hR$df), eR), relgap(c(hI$est, hI$se[se], hI$df), eI))
      out[[se]] <- list(REG = list(est = unname(a$coefficients["w"]), se = unname(a$std.error["w"]),
                                   df = unname(a$df["w"]), ci = unname(c(a$conf.low["w"], a$conf.high["w"]))),
                        IREG = list(est = unname(l$coefficients["w"]), se = unname(l$std.error["w"]),
                                    df = unname(l$df["w"]), ci = unname(c(l$conf.low["w"], l$conf.high["w"]))))
    }
    n_checked <- n_checked + 1
    if (tag == "" && b <= 5) {
      stem <- file.path(ROOT, "results", "r_datasets", sprintf("%s_%02d", dn, b))
      write.csv(df, paste0(stem, ".csv"), row.names = FALSE)
      out$leverage_IREG <- hand_hc(z$y, MI)$h
      write_json(out, paste0(stem, ".json"), auto_unbox = TRUE, digits = NA)
    }
  }
}
report$formula_vs_engine_max_rel_gap <- gap_formula
report$hand_vs_estimatr_max_rel_gap <- gap_hand
report$datasets_checked <- n_checked

# ------------------------------------------------ 4: gamma = 0
d <- list(x = "lognormal", s = 1.2, het = "prop", mis = "none", p = 1, n = 500)
z <- draw(d)
Xc <- sweep(z$X, 2, colMeans(z$X)); MI <- cbind(1, z$w, Xc, z$w * Xc)
bI <- qr.coef(qr(MI), z$y); delta <- bI[3 + d$p - 1 + seq_len(d$p)]
report$gamma0 <- list(population_G = 0, max_abs_ite_minus_tau = max(abs(z$ite - 1)),
                      sate_minus_pate = mean(z$ite) - 1,
                      plugin_deltaSigmadelta_over_n = drop(t(delta) %*% cov(z$X) %*% delta) / d$n)

# ------------------------------------------------ 5: p = 1 closed forms
p1 <- 0
for (b in 1:20) {
  z <- draw(list(x = "lognormal", s = 1.0, het = "prop", mis = "none", p = 1, n = 200))
  x <- z$X[, 1]; w <- z$w; y <- z$y
  st <- function(k) { xx <- x[w == k]; yy <- y[w == k]
    c(xb = mean(xx), yb = mean(yy), S = sum((xx - mean(xx))^2), s = sum((xx - mean(xx)) * (yy - mean(yy)))) }
  s1 <- st(1); s0 <- st(0); D <- s1["xb"] - s0["xb"]
  reg <- s1["yb"] - s0["yb"] - D * (s1["s"] + s0["s"]) / (s1["S"] + s0["S"])
  ireg <- s1["yb"] - s0["yb"] - D * (s1["s"] / s1["S"] + s0["s"] / s0["S"]) / 2
  eR <- fit_one(y, cbind(1, w, x), "HC1"); xc <- x - mean(x); eI <- fit_one(y, cbind(1, w, xc, w * xc), "HC1")
  p1 <- max(p1, relgap(unname(c(reg, ireg)), unname(c(eR["est"], eI["est"]))))
}
report$p1_closed_form_max_rel_gap <- p1

# ------------------------------------------------ 6: leverage one
q <- 0.02; n <- 500; m <- n / 2
repeat {
  x <- ifelse(runif(n) < q, sqrt((1 - q) / q), -sqrt(q / (1 - q))); w <- sample(rep(c(1, 0), c(m, n - m)))
  if (sum(x > 0 & w == 1) == 1 && sum(x > 0 & w == 0) >= 2) break
}
y <- 0.9 * x + w + (1 + abs(x)) * rnorm(n)
xc <- x - mean(x); MI <- cbind(1, w, xc, w * xc)
hh <- hand_hc(y, MI); i <- which(x > 0 & w == 1)
A <- solve(crossprod(MI)); cvec <- (A %*% t(MI))[2, ]
se_zeroed <- function(pow) { wts <- hh$e^2 / (1 - hh$h)^pow; wts[i] <- 0; sqrt(sum(cvec^2 * wts)) }
warn <- character(0)
fits <- withCallingHandlers(
  lapply(c("HC1", "HC2", "HC3"), function(se) lm_lin(y ~ w, covariates = ~x, data = data.frame(y, w, x), se_type = se)),
  warning = function(wn) { warn <<- c(warn, conditionMessage(wn)); invokeRestart("muffleWarning") })
report$leverage_one <- list(
  unit = i, leverage = hh$h[i], one_minus_leverage = 1 - hh$h[i], residual = hh$e[i],
  estimatr_se = setNames(lapply(fits, function(f) unname(f$std.error["w"])), c("HC1", "HC2", "HC3")),
  se_with_unit_contribution_zeroed = list(HC2 = se_zeroed(1), HC3 = se_zeroed(2)),
  estimatr_warnings = unique(warn))

dir.create(file.path(ROOT, "results"), showWarnings = FALSE)
write_json(report, file.path(ROOT, "results", paste0("r_validation", tag, ".json")),
           auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat(toJSON(report[c("estimatr_version", "formula_vs_engine_max_rel_gap", "hand_vs_estimatr_max_rel_gap",
                    "datasets_checked", "p1_closed_form_max_rel_gap")], auto_unbox = TRUE), "\n")
cat(toJSON(report$gamma0, auto_unbox = TRUE, digits = NA), "\n")
cat(toJSON(report$leverage_one, auto_unbox = TRUE, digits = NA), "\n")
stopifnot(gap_formula < 1e-8, gap_hand < 1e-8, p1 < 1e-8, report$gamma0$max_abs_ite_minus_tau < 1e-10)
