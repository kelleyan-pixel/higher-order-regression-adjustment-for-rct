#!/usr/bin/env Rscript
# Transformation robustness check (Section 3.3): on exactly the datasets of the gamma = 0,
# s = 2 design of Table 1 (same seed and draw order as r/gamma0_sim.R), compare REG and IREG
# adjusting for the raw covariate X with the same estimators adjusting for Z = log(1 + X).
# The outcome model is unchanged (linear in X), so adjusting for Z is a deliberately
# misspecified but still valid adjustment under randomization.
#
# Usage: Rscript r/transform_check.R
# Output: results/r_transform_check.json
args <- commandArgs(trailingOnly = FALSE)
script <- normalizePath(sub("^--file=", "", args[grep("^--file=", args)]))
ROOT <- dirname(dirname(script))
suppressMessages(library(estimatr)); suppressMessages(library(jsonlite))
source(file.path(ROOT, "r", "common.R"))

group <- "skew"; i <- 7L
d <- design_list(group)[[i]]
stopifnot(d$s == 2.0)
seed <- 20260921L + 1000L * match(group, c("skew", "misspec", "nsweep", "p5", "rare", "smoke")) + i
set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
R <- d$reps
cols <- c("est", "se", "df")
out <- array(NA_real_, c(R, 2, 4, 3), dimnames = list(NULL, c("X", "log1pX"),
             c("REG_HC1", "REG_HC3", "IREG_HC1", "IREG_HC3"), cols))
lev <- matrix(NA_real_, R, 2, dimnames = list(NULL, c("X", "log1pX")))
for (b in seq_len(R)) {
  z <- draw(d)
  for (cv in c("X", "log1pX")) {
    C <- if (cv == "X") z$X else log1p(z$X)
    MR <- cbind(1, z$w, C); Cc <- sweep(C, 2, colMeans(C)); MI <- cbind(1, z$w, Cc, z$w * Cc)
    for (se in c("HC1", "HC3")) {
      out[b, cv, paste0("REG_", se), ] <- fit_one(z$y, MR, se)
      out[b, cv, paste0("IREG_", se), ] <- fit_one(z$y, MI, se)
    }
    lev[b, cv] <- leverage(MI)
  }
}
res <- list(design = d, seed = seed, estimatr_version = as.character(packageVersion("estimatr")), reps = R)
for (cv in c("X", "log1pX")) {
  r <- list(median_max_leverage_IREG = median(lev[, cv], na.rm = TRUE))
  for (m in dimnames(out)[[3]]) {
    est <- out[, cv, m, "est"]; se <- out[, cv, m, "se"]; df <- out[, cv, m, "df"]
    ok <- is.finite(est) & is.finite(se) & is.finite(df)
    q <- qt(0.975, pmax(df, 1))
    hit <- ok & abs(est - 1) <= q * se
    cvg <- mean(hit)
    r[[m]] <- list(coverage = cvg, mcse = sqrt(cvg * (1 - cvg) / R), median_width = median(2 * q[ok] * se[ok]))
  }
  res[[cv]] <- r
}
write_json(res, file.path(ROOT, "results", "r_transform_check.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat(sprintf("X:       IREG HC1 %.4f  IREG HC3 %.4f  REG HC1 %.4f  REG HC3 %.4f\n", res$X$IREG_HC1$coverage,
            res$X$IREG_HC3$coverage, res$X$REG_HC1$coverage, res$X$REG_HC3$coverage))
cat(sprintf("log1pX:  IREG HC1 %.4f  IREG HC3 %.4f  REG HC1 %.4f  REG HC3 %.4f\n", res$log1pX$IREG_HC1$coverage,
            res$log1pX$IREG_HC3$coverage, res$log1pX$REG_HC1$coverage, res$log1pX$REG_HC3$coverage))
