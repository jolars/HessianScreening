library(HessianScreening)

printf <- function(...) invisible(cat(sprintf(...)))

tol_gap <- 1e-4
families <- c("gaussian")
scenarios <- c(2)
rhos <- c(0, 0.4, 0.8)
n <- 200
p <- 20000
snr <- 2
s <- 20

screening_types <- c(
  "hessian",
  "working"
)
family <- "gaussian"

path_length <- 100

n_sim <- length(families) * length(rhos)
out <- data.frame()

max_it <- 20

it_sim <- 0

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

    for (screening_type in screening_types) {
      for (gap_safe in c(TRUE, FALSE)) {
        printf(
          "\r%s, it: %02d/%02d %-17s",
          format(Sys.time(), "%H:%M:%S"),
          i,
          max_it,
          screening_type
        )
        flush.console()

        set.seed(i)

        fit <- lassoPath(
          X,
          y,
          family = family,
          lambda = lambda,
          screening_type = screening_type,
          path_length = path_length,
          log_hessian_update_type = "full",
          augment_with_gap_safe = gap_safe,
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
          screening_type = screening_type,
          gap_safe = if (gap_safe) "With Gap Safe" else "Without Gap Safe",
          rho = rho,
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

saveRDS(out, "results/heuristic-with-or-without-gap-safe.rds")
