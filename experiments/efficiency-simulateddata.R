library(HessianScreening)

printf <- function(...) invisible(cat(sprintf(...)))

tol_gap <- 1e-4
families <- c("gaussian", "binomial")
rhos <- c(0, 0.2)
n <- 200
p <- 20000
snr <- 2
s <- 20

screening_types <- c(
  "hessian",
  # "edpp",
  # "gap_safe",
  # "strong",
  # "celer",
  "blitz"
)

path_length <- 100

n_sim <- length(families) * length(rhos)
out <- data.frame()

max_it <- 50

it_sim <- 0

for (family in families) {
  for (rho in rhos) {
    it_sim <- it_sim + 1

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
        screening_type = "hessian",
        path_length = path_length,
        verbosity = 0,
        line_search = TRUE,
        tol_gap = tol_gap
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
          rho = rho,
          it = i,
          screening_type = screening_type,
          screened = fit$screened,
          active = fit$active,
          violations = fit$violations,
          converged = all(fit$converged)
        )

        out <- rbind(out, res)
      }
    }
  }
}

cat("\n")

saveRDS(out, "results/efficiency-simulateddata.rds")
