# Shared data-generating processes and fitting helpers for the R simulations.
# Sourced by r/gamma0_sim.R and r/validate_estimatr.R.
LEV_TOL <- 1e-8

# ------------------------------------------------------------------ designs
design_list <- function(group) {
  D <- function(name, x, s = NA, het, mis = "none", p = 1, n = 500, reps = 20000, q = NA)
    list(name = name, x = x, s = s, het = het, mis = mis, p = p, n = n, reps = reps, q = q)
  switch(group,
    skew = lapply(c(0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0), function(s)
      D(sprintf("lognormal_s%.1f_prop", s), "lognormal", s = s, het = "prop")),
    misspec = list(
      D("gauss_quad", "gauss", het = "none", mis = "quad_gauss"),
      D("lognormal_s1.0_log", "lognormal", s = 1.0, het = "none", mis = "log"),
      D("lognormal_s1.0_quadstd", "lognormal", s = 1.0, het = "none", mis = "quad_std"),
      D("lognormal_s1.0_log_prop", "lognormal", s = 1.0, het = "prop", mis = "log")),
    nsweep = list(
      D("lognormal_s1.2_prop_n200", "lognormal", s = 1.2, het = "prop", n = 200),
      D("lognormal_s1.2_prop_n2000", "lognormal", s = 1.2, het = "prop", n = 2000, reps = 10000),
      D("lognormal_s1.2_prop_n8000", "lognormal", s = 1.2, het = "prop", n = 8000, reps = 4000)),
    p5 = list(
      D("lognormal_s1.0_prop_p5_n500", "lognormal", s = 1.0, het = "prop", p = 5, n = 500),
      D("lognormal_s1.0_prop_p5_n200", "lognormal", s = 1.0, het = "prop", p = 5, n = 200)),
    rare = list(
      D("rare_q0.02_prop", "rare", q = 0.02, het = "prop"),
      D("rare_q0.02_homo", "rare", q = 0.02, het = "none")),
    smoke = list(D("smoke", "lognormal", s = 1.0, het = "prop", reps = 200)),
    stop("unknown group"))
}

draw <- function(d) {
  n <- d$n; p <- d$p; m <- n %/% 2
  X <- matrix(rnorm(n * p), n, p)
  if (d$x == "lognormal") X[, 1] <- exp(d$s * X[, 1])
  if (d$x == "rare") {
    q <- d$q
    X[, 1] <- ifelse(runif(n) < q, sqrt((1 - q) / q), -sqrt(q / (1 - q)))
  }
  x1 <- X[, 1]
  sig <- if (d$het == "prop") 1 + abs(x1) else rep(1, n)
  g <- switch(d$mis,
    none = 0,
    quad_gauss = 0.3 * (x1^2 - 1),
    log = 2 * log(x1),
    quad_std = {
      mu <- exp(d$s^2 / 2); sdv <- sqrt((exp(d$s^2) - 1) * exp(d$s^2))
      0.3 * (((x1 - mu) / sdv)^2 - 1)
    })
  w <- sample(rep(c(1, 0), c(m, n - m)))
  eps <- rnorm(n)
  y0 <- drop(X %*% rep(0.9, p)) + g + sig * eps       # gamma = 0: Y(1) - Y(0) = tau = 1
  y1 <- y0 + 1
  list(X = X, w = w, y = ifelse(w == 1, y1, y0), ite = y1 - y0)
}

fit_one <- function(y, M, se_type) {
  r <- estimatr:::lm_robust_fit(y = y, X = M, weights = NULL, cluster = NULL, ci = TRUE,
                                se_type = se_type, has_int = TRUE, alpha = 0.05,
                                return_vcov = FALSE, try_cholesky = FALSE)
  c(est = unname(r$coefficients[2]), se = unname(r$std.error[2]), df = unname(r$df[2]))
}

leverage <- function(M) {
  qm <- qr(M)
  if (qm$rank < ncol(M)) return(NA_real_)
  max(rowSums(qr.Q(qm)^2))
}

