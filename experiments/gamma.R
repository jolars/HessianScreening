library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

printf <- function(...) invisible(cat(sprintf(...)))

n <- 400
p <- 40000
snr <- 2
s <- 20
tol_gap <- 1e-4
path_length <- 100
families <- c("gaussian")
rhos <- c(0, 0.4, 0.8)
gammas <- 10^seq(log10(0.001), log10(0.3), length.out = 20)

n_sim <- length(rhos)
out <- data.frame()

max_it <- 50

it_sim <- 0

for (rho in rhos) {
  it_sim <- it_sim + 1

  printf("\r%02d/%i rho: %0.2f\n", it_sim, n_sim, rho)

  for (i in 1:max_it) {
    set.seed(i)

    d <- generateDesign(n, p, family = "gaussian", rho = rho, snr = snr)
    X <- d$X
    y <- d$y

    fit <- lassoPath(
      X,
      y,
      screening_type = "hessian",
      path_length = path_length
    )

    lambda <- fit$lambda

    for (gamma in gammas) {
      printf(
        "\r%s, it: %02d, gamma: %0.4f",
        format(Sys.time(), "%H:%M:%S"),
        i,
        gamma
      )
      flush.console()

      fit <- lassoPath(
        X,
        y,
        screening_type = "hessian",
        lambda = lambda,
        gamma = gamma
      )

      n_lambda <- length(fit$lambda)

      res <- data.frame(
        rho = rho,
        gamma = gamma,
        it = i,
        step = seq_along(fit$full_time),
        time = fit$full_time,
        active = fit$active,
        screened = fit$screened,
        violations = fit$violations,
        converged = all(fit$converged)
      )

      out <- rbind(out, res)
    }
    cat("\n")
  }
}

cat("\n")

saveRDS(out, "results/gamma.rds")
