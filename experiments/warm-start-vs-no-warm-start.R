library(HessianScreening)

printf <- function(...) invisible(cat(sprintf(...)))

tol_gap <- 1e-4
rhos <- c(0, 0.4, 0.8)
families <- c("gaussian", "binomial")
scenarios <- c(1, 2)
rhos <- 0.3

methods <- c(
  "hessian",
  "hessian_warmstart",
  "working"
)

path_length <- 100

n_sim <- length(families) * length(methods) * length(scenarios)
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
      n <- 400
      p <- 40000
      snr <- 2
      s <- 20
    }

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
          tol_gap = 1e-4
        )

        lambda <- fit$lambda

        for (method in methods) {
          if (method %in% c("hessian", "hessian_warmstart")) {
            screening_type <- "hessian"

            hessian_warm_starts <- method == "hessian_warmstart"
          } else {
            screening_type <- "working"
            hessian_warm_starts <- FALSE
          }
          printf(
            "\r%s, it: %02d/%02d %-10s",
            format(Sys.time(), "%H:%M:%S"),
            i,
            max_it,
            method
          )
          flush.console()

          fit <- lassoPath(
            X,
            y,
            family = family,
            lambda = lambda,
            screening_type = screening_type,
            hessian_warm_starts = hessian_warm_starts,
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
            method = method,
            it = i,
            time = fit$full_time,
            passes = fit$passes,
            converged = all(fit$converged)
          )

          out <- rbind(out, res)
        }
      }
    }
  }
}

cat("\n")

saveRDS(out, "results/warm-start-vs-no-warm-start.rds")
