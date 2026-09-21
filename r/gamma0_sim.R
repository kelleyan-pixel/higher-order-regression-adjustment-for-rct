#!/usr/bin/env Rscript
# Finite-sample coverage of REG and IREG (lm_lin) intervals when gamma = 0.
#
# Every design has beta1 = beta0 and a constant individual treatment effect
# (tau = 1, shared residual noise), so R^2_tau = 0 exactly and the population and
# sample ATEs coincide.  Any undercoverage is therefore a finite-sample failure of
# the variance estimator, not an estimand effect.
#
# Fits use estimatr's own fitting routine (estimatr:::lm_robust_fit, the function
# that lm_robust() and lm_lin() call after building the design matrix), with the
# design matrices built exactly as lm_robust(y ~ w + x) and lm_lin(y ~ w, ~ x) build
# them.  r/validate_estimatr.R checks that this reproduces the formula interfaces
# exactly.  Confidence intervals use t critical values with estimatr's reported df.
#
# A replication "covers" only if estimatr returns a finite estimate, standard error
# and df; NaN output (e.g. HC2 at a computed leverage slightly above one under
# estimatr 1.x) counts as a non-cover and is reported in frac_no_interval.
#
# Usage (from anywhere):
#   Rscript r/gamma0_sim.R <group> [--lib=/path/to/alternative/R/library] [--tag=suffix]
#   (--reps=N overrides the replication counts; used only for quick pipeline tests)
# Groups: skew, misspec, nsweep, p5, rare, smoke
# Output: results/r_gamma0_<group><tag>.json

args <- commandArgs(trailingOnly = FALSE)
script <- normalizePath(sub("^--file=", "", args[grep("^--file=", args)]))
ROOT <- dirname(dirname(script))
targs <- commandArgs(trailingOnly = TRUE)
group <- targs[1]
libarg <- sub("^--lib=", "", targs[grep("^--lib=", targs)])
tag <- sub("^--tag=", "", targs[grep("^--tag=", targs)])
if (length(tag) == 0) tag <- ""
repsarg <- sub("^--reps=", "", targs[grep("^--reps=", targs)])   # testing only: override replication counts
if (length(libarg) == 1) .libPaths(c(libarg, .libPaths()))
suppressMessages(library(estimatr))
suppressMessages(library(jsonlite))
source(file.path(ROOT, "r", "common.R"))

run_design <- function(d, seed) {
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  R <- d$reps
  out <- matrix(NA_real_, R, 21)
  colnames(out) <- c(paste0("R_", c("est1", "se1", "df1", "est2", "se2", "df2", "est3", "se3", "df3")),
                     paste0("I_", c("est1", "se1", "df1", "est2", "se2", "df2", "est3", "se3", "df3")),
                     "levI", "rankI", "maxite_dev")
  for (b in seq_len(R)) {
    z <- draw(d)
    MR <- cbind(1, z$w, z$X)                                  # lm_robust(y ~ w + x)
    Xc <- sweep(z$X, 2, colMeans(z$X))
    MI <- cbind(1, z$w, Xc, z$w * Xc)                         # lm_lin(y ~ w, ~ x)
    rr <- c(); ri <- c()
    for (se in c("HC1", "HC2", "HC3")) {
      rr <- c(rr, fit_one(z$y, MR, se)); ri <- c(ri, fit_one(z$y, MI, se))
    }
    out[b, ] <- c(rr, ri, leverage(MI), qr(MI)$rank, max(abs(z$ite - 1)))
  }
  out
}

summarize <- function(d, out) {
  full <- out[, "rankI"] == 2 + 2 * d$p
  lev1 <- full & !is.na(out[, "levI"]) & out[, "levI"] > 1 - LEV_TOL
  cov_stats <- function(pre, k, mask) {
    est <- out[, paste0(pre, "_est", k)]; se <- out[, paste0(pre, "_se", k)]
    df <- out[, paste0(pre, "_df", k)]
    ok <- is.finite(est) & is.finite(se) & is.finite(df)
    hit <- ok & abs(est - 1) <= qt(0.975, pmax(df, 1)) * se
    N <- sum(mask)
    cc <- if (N > 0) mean(hit[mask]) else NA
    list(coverage = cc, mcse = if (N > 0) sqrt(cc * (1 - cc) / N) else NA, n = N,
         frac_no_interval = if (N > 0) mean(!ok[mask]) else NA)
  }
  res <- list(design = d, reps = nrow(out),
              frac_rank_deficient_IREG = mean(!full),
              frac_no_interval_IREG = mean(!is.finite(out[, "I_est3"]) | !is.finite(out[, "I_se3"])),
              frac_leverage_one_IREG = mean(lev1),
              median_max_leverage_IREG = median(out[full, "levI"]),
              max_abs_ite_minus_tau = max(out[, "maxite_dev"]),
              estimatr_version = as.character(packageVersion("estimatr")))
  for (pre in c("R", "I")) for (k in 1:3) {
    lab <- paste0(ifelse(pre == "R", "REG", "IREG"), "_HC", k)
    res[[lab]] <- list(all = cov_stats(pre, k, rep(TRUE, nrow(out))),
                       full_rank = cov_stats(pre, k, full),
                       leverage_one = cov_stats(pre, k, lev1))
  }
  res
}

designs <- design_list(group)
if (length(repsarg) == 1) designs <- lapply(designs, function(d) { d$reps <- as.integer(repsarg); d })
results <- list()
t0 <- Sys.time()
for (i in seq_along(designs)) {
  d <- designs[[i]]
  seed <- 20260921L + 1000L * match(group, c("skew", "misspec", "nsweep", "p5", "rare", "smoke")) + i
  out <- run_design(d, seed)
  s <- summarize(d, out); s$seed <- seed
  results[[d$name]] <- s
  cat(sprintf("%-32s IREG HC1 %.4f HC2 %.4f HC3 %.4f | REG HC1 %.4f HC3 %.4f | rankdef %.4f lev1 %.4f  [%.0fs]\n",
              d$name, s$IREG_HC1$all$coverage, s$IREG_HC2$all$coverage, s$IREG_HC3$all$coverage,
              s$REG_HC1$all$coverage, s$REG_HC3$all$coverage,
              s$frac_rank_deficient_IREG, s$frac_leverage_one_IREG,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
dir.create(file.path(ROOT, "results"), showWarnings = FALSE)
write_json(results, file.path(ROOT, "results", paste0("r_gamma0_", group, tag, ".json")),
           auto_unbox = TRUE, digits = NA, pretty = TRUE)
