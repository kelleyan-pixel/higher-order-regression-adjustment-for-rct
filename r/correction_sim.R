#!/usr/bin/env Rscript
# Diagnostic: standard vs superpopulation-corrected IREG intervals on the gamma = 0
# designs of Section 3 (paper Tables 1-3).  Reuses the data-generating processes,
# seeds, estimatr fits and draw order of r/gamma0_sim.R exactly, so the uncorrected
# coverages reproduce results/r_gamma0_*.json; the only addition is the plug-in
#   delta_hat' Sigma_hat_X delta_hat / n
# computed from the IREG interaction coefficients of the same fit.  Rank-deficient
# fits (a dropped interaction column, NA coefficient) use 0 for the dropped entry.
#
# Usage: Rscript r/correction_sim.R [group ...]      (default: all groups)
# Output: results/correction/r_<design>.json, one file per design (resumable)
args <- commandArgs(trailingOnly = FALSE)
script <- normalizePath(sub("^--file=", "", args[grep("^--file=", args)]))
ROOT <- dirname(dirname(script))
suppressMessages(library(estimatr)); suppressMessages(library(jsonlite))
source(file.path(ROOT, "r", "common.R"))
groups <- commandArgs(trailingOnly = TRUE)
if (length(groups) == 0) groups <- c("skew", "misspec", "nsweep", "p5", "rare")
outdir <- file.path(ROOT, "results", "correction"); dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

fit_full <- function(y, M, se_type) {
  r <- estimatr:::lm_robust_fit(y = y, X = M, weights = NULL, cluster = NULL, ci = TRUE,
                                se_type = se_type, has_int = TRUE, alpha = 0.05,
                                return_vcov = FALSE, try_cholesky = FALSE)
  list(v = c(est = unname(r$coefficients[2]), se = unname(r$std.error[2]), df = unname(r$df[2])),
       coef = unname(r$coefficients))
}

for (group in groups) {
  designs <- design_list(group)
  for (i in seq_along(designs)) {
    d <- designs[[i]]
    f <- file.path(outdir, paste0("r_", d$name, ".json"))
    if (file.exists(f)) { cat("skip", d$name, "\n"); next }
    seed <- 20260921L + 1000L * match(group, c("skew", "misspec", "nsweep", "p5", "rare", "smoke")) + i
    set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    R <- d$reps; p <- d$p; n <- d$n
    out <- matrix(NA_real_, R, 21)
    colnames(out) <- c(paste0("R_", c("est1","se1","df1","est2","se2","df2","est3","se3","df3")),
                       paste0("I_", c("est1","se1","df1","est2","se2","df2","est3","se3","df3")),
                       "plug", "levI", "rankI")
    t0 <- Sys.time()
    for (b in seq_len(R)) {
      z <- draw(d)
      MR <- cbind(1, z$w, z$X); Xc <- sweep(z$X, 2, colMeans(z$X)); MI <- cbind(1, z$w, Xc, z$w * Xc)
      rr <- c(); ri <- c(); delta <- NULL
      for (se in c("HC1", "HC2", "HC3")) {
        a <- fit_full(z$y, MR, se); g <- fit_full(z$y, MI, se)
        rr <- c(rr, a$v); ri <- c(ri, g$v)
        if (is.null(delta)) { delta <- g$coef[2 + p + seq_len(p)]; delta[is.na(delta)] <- 0 }
      }
      plug <- drop(t(delta) %*% cov(z$X) %*% delta) / n
      out[b, ] <- c(rr, ri, plug, leverage(MI), qr(MI)$rank)
    }
    write_json(list(design = d, seed = seed, estimatr_version = as.character(packageVersion("estimatr")),
                    G = 0, reps = R, sims = as.data.frame(out)),
               f, auto_unbox = TRUE, digits = NA)
    cat(sprintf("%-32s done in %.0fs\n", d$name, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
}
