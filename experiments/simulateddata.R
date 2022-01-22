library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

printf <- function(...) invisible(cat(sprintf(...)))

# g <- expand_grid(
#   family = c("gaussian", "binomial"),
#   scenario = c(1, 2),
#   tol_gap = c(1e-4),
#   n = NA,
#   p = NA,
#   rho = c(0, 0.4, 0.8),
#   screening_type = c(
#     "hessian",
#     "working",
#     # "edpp",
#     # "gap_safe",
#     "blitz",
#     "celer"
#   ),
#   path_length = 100,
#   avg_screened = NA,
#   avg_violations = NA,
#   time = list(NA),
#   screened = list(NA),
#   active = list(NA),
#   step = list(NA),
#   converged = NA
# )

tol_gap <- 1e-4

families <- c("gaussian", "binomial")

scenarios <- c(1, 2)

rhos <- c(0, 0.4, 0.8)

screening_types <- c(
  "hessian",
  "working",
  # "edpp",
  # "gap_safe",
  "blitz",
  "celer"
)

path_length <- 100

n_sim <- length(families) * length(scenarios) * length(rhos)

out <- data.frame()

max_it <- 5

it_sim <- 0L

for (family in families) {
  for (scenario in scenarios) {
    if (scenario == 1) {
      n <- 10000
      p <- 100
      snr <- 1
      s <- 5
    } else if (scenario == 2) {
      n <- 200
      p <- 20000
      snr <- 2
      s <- 20
    }

    for (rho in rhos) {
      it_sim <- it_sim + 1L

      printf(
        "\r%02d/%i %-10s n: %4d p: %4d rho: %0.2f\n",
        it_sim, n_sim, family, n, p, rho
      )

      for (i in 1:max_it) {
        set.seed(i)

        d <- generateDesign(n, p, family = family, rho = rho, snr = snr)
        X <- d$X
        y <- d$y

        fit <- lassoPath(
          X,
          y,
          family = family,
          screening_type = "working",
          path_length = path_length,
          verbosity = 0,
          tol_gap = tol_gap,
          line_search = 1
        )

        lambda <- fit$lambda

        for (screening_type in screening_types) {
          printf(
            "\r%s, it: %02d %-10s",
            format(Sys.time(), "%H:%M:%S"),
            i,
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
            tol_gap = tol_gap,
            line_search = 1
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
            rho = rho,
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
}



cat("\n")

saveRDS(g, "results/simulateddata.rds")
