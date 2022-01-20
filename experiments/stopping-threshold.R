library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

printf <- function(...) invisible(cat(sprintf(...)))

g <- expand_grid(
  family = c("gaussian", "binomial"),
  scenario = c(1, 2),
  n = NA,
  p = NA,
  rho = c(0),
  tol_gap = c(1e-4, 1e-5, 1e-6, 1e-7),
  screening_type = c(
    "hessian",
    "working",
    "blitz",
    "celer"
  ),
  path_length = 100,
  time = list(NA),
  step = list(NA),
  converged = NA
)

min_it <- 10
max_it <- 100 * min_it
max_err <- 0.1
conf_level <- 0.05

for (i in seq_len(nrow(g))) {
  rho <- g$rho[i]
  family <- g$family[i]
  screening_type <- g$screening_type[i]
  path_length <- g$path_length[i]
  scenario <- g$scenario[i]
  tol_gap <- g$tol_gap[i]

  if (family == "binomial" && screening_type == "edpp") {
    next
  }

  if (scenario == 1) {
    n <- 10000
    p <- 100
    snr <- 1
    s <- 5
  } else if (scenario == 2) {
    n <- 400
    p <- 40000
    snr <- 2
    s <- 20
  }

  printf(
    "%02d/%i %-10s n: %4d p: %4d tol_gap: %1.1e %-10s\n",
    i, nrow(g), family, n, p, tol_gap, screening_type
  )

  time <- double(max_it)

  for (j in seq_len(max_it)) {
    set.seed(j)

    d <- generateDesign(n, p, family = family, rho = rho, snr = snr)
    X <- d$X
    y <- d$y

    fit <- lassoPath(
      X,
      y,
      family = family,
      screening_type = screening_type,
      path_length = path_length,
      log_hessian_update_type = "full",
      verbosity = 0,
      tol_gap = tol_gap
    )

    n_lambda <- length(fit$lambda)

    time[j] <- fit$full_time

    if (any(!fit$converged)) {
      warning(
        "failed to converge at i = ",
        i,
        " for solver = ",
        screening_type,
        " at steps ",
        paste(which(!fit$converged)),
        collapse = ","
      )
    }

    # stop if standard error is within 2.5% of mean
    if (j > min_it) {
      se <- sd(time[1:j]) / sqrt(j)
      ci_width <- 2 * qt(1 - conf_level / 2, df = j - 1) * se

      if (ci_width / mean(time[1:j]) < max_err) {
        break
      }
    }
  }

  time <- time[1:j]

  g$n[i] <- n
  g$p[i] <- p
  g$family[i] <- family
  g$time[i] <- list(time)
  g$step[i] <- list(1:path_length)
  g$converged[i] <- all(fit$converged)
}

saveRDS(g, "results/stopping-threshold.rds")
