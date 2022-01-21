library(HessianScreening) 
library(tibble)
library(dplyr)
library(tidyr)

printf <- function(...) invisible(cat(sprintf(...)))

g <- expand_grid(
  family = c("gaussian", "binomial"),
  scenario = c(1, 2),
  tol_gap = c(1e-6),
  n = NA,
  p = NA,
  rho = c(0, 0.4, 0.8),
  screening_type = c(
    "hessian",
    "working",
    # "edpp",
    # "gap_safe",
    "blitz",
    "celer"
  ),
  path_length = 100,
  avg_screened = NA,
  avg_violations = NA,
  time = list(NA),
  screened = list(NA),
  active = list(NA),
  step = list(NA),
  converged = NA
)

min_it <- 10
max_it <- 1000
max_err <- 0.2
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

  avg_screened <- violations <- time <- double(max_it)
  active <- screened <- matrix(NA, nrow = max_it, ncol = path_length)

  printf(
    "\r%02d/%i %-10s n: %4d p: %4d rho: %1.1f %-10s\n",
    i, nrow(g), family, n, p, rho, screening_type
  )

  for (j in seq_len(max_it)) {
    set.seed(j)

    printf("\r%s, it: %02d", format(Sys.time(), "%H:%M:%S"), j)
    flush.console() 

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
    avg_screened[j] <- mean(fit$active / fit$screened)
    violations[j] <- sum(fit$violations)
    screened[j, 1:n_lambda] <- fit$screened
    active[j, 1:n_lambda] <- fit$active

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
    if (j >= min_it) {
      se <- sd(time[1:j]) / sqrt(j)
      ci_width <- 2 * qnorm(1 - conf_level / 2) * se

      if (ci_width / mean(time[1:j]) < max_err) {
        break
      }
    }
  }

  time <- time[1:j]
  active <- active[1:j, ]
  screened <- screened[1:j, ]
  violations <- violations[1:j]
  avg_screened <- avg_screened[1:j]

  dontuse <- apply(screened, 2, anyNA)

  active <- colMeans(active)
  screened <- colMeans(screened)

  g$n[i] <- n
  g$p[i] <- p
  g$family[i] <- family
  g$time[i] <- list(time)
  g$avg_screened[i] <- mean(avg_screened)
  g$avg_violations[i] <- mean(violations)
  g$screened[i] <- list(screened)
  g$active[i] <- list(active)
  g$step[i] <- list(1:path_length)
  g$converged[i] <- all(fit$converged)
}

cat("\n")

saveRDS(g, "results/simulateddata.rds")
