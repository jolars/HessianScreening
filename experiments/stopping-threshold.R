library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

printf <- function(...) invisible(cat(sprintf(...)))

families <- c("gaussian", "binomial")
rho <- 0.4

n <- 200
p <- 20000
snr <- 2
s <- 20

tol_gaps <- c(1e-3, 1e-4, 1e-5, 1e-6)

screening_types <- c(
  "hessian",
  "working",
  # "edpp",
  # "gap_safe",
  "blitz",
  "celer"
)

path_length <- 100

n_sim <- length(families)
out <- data.frame()

max_it <- 20

it_sim <- 0

for (family in families) {
  it_sim <- it_sim + 1

  printf("\r%02d/%i %-10s\n", it_sim, n_sim, family)

  for (i in 1:max_it) {
    set.seed(i)

    d <- generateDesign(n, p, family = family, rho = rho, snr = snr)
    X <- d$X
    y <- d$y

    fit <- lassoPath(
      X,
      y,
      family = family,
      screening_type = "hessian",
      path_length = path_length,
      verbosity = 0,
      line_search = TRUE,
      tol_gap = 1e-6
    )

    lambda <- fit$lambda

    for (tol_gap in tol_gaps) {
      it_sim <- it_sim + 1

      for (screening_type in screening_types) {
        printf(
          "\r%s, it: %02d/%02d tol_gap: %1.1e %-10s",
          format(Sys.time(), "%H:%M:%S"),
          i,
          max_it,
          tol_gap,
          screening_type
        )
        flush.console()

        fit <- lassoPath(
          X,
          y,
          family = family,
          lambda = lambda,
          screening_type = screening_type,
          path_length = path_length,
          log_hessian_update_type = "full",
          celer_prune = TRUE,
          verbosity = 0,
          tol_gap = tol_gap
        )

        n_lambda <- length(fit$lambda)

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

        res <- data.frame(
          family = family,
          n = n,
          p = p,
          tol_gap = tol_gap,
          screening_type = screening_type,
          it = i,
          time = fit$full_time,
          converged = all(fit$converged)
        )

        out <- rbind(out, res)
      }
    }
  }
}

cat("\n")

saveRDS(out, "results/stopping-threshold.rds")
