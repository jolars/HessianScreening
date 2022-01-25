library(HessianScreening)

printf <- function(...) invisible(cat(sprintf(...)))

tol_gap <- 1e-4
families <- c("gaussian", "binomial")
scenarios <- c(1, 2)
rho <- 0.4

screening_types <- c(
  "hessian",
  "working",
  # "edpp",
  # "gap_safe",
  "blitz",
  "celer"
)

path_lengths <- c(10, 20, 50, 100, 200)

n_sim <- length(families) * length(path_lengths) * length(scenarios) 
out <- data.frame()

max_it <- 20

it_sim <- 0

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

    for (path_length in path_lengths) {
      it_sim <- it_sim + 1

      printf(
        "\r%02d/%i %-10s n: %4d p: %4d path_length: %3d\n",
        it_sim, n_sim, family, n, p, path_length
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
            n = n,
            p = p,
            path_length = path_length,
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

saveRDS(out, "results/path-length.rds")
