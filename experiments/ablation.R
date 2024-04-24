library(HessianScreening)

printf <- function(...) invisible(cat(sprintf(...)))

tol_gap <- 1e-4
scenarios <- c(2)
rhos <- c(0, 0.4)
n <- 200
p <- 20000
snr <- 2
s <- 20

family <- "gaussian"

path_length <- 100

n_sim <- length(rhos) * 5
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

    # fit <- lassoPath(
    #   X,
    #   y,
    #   family = family,
    #   screening_type = "none",
    #   hessian_warm_starts = FALSE,
    #   update_hessian = FALSE,
    #   path_length = path_length,
    #   verbosity = 2,
    #   line_search = TRUE,
    #   tol_gap = 1e-4
    # )

    lambda <- fit$lambda

    for (ablation in 1:5) {
      screening_type <- if (ablation > 1) "hessian" else "none"
      hessian_warm_starts <- ablation > 2
      update_hessian <- ablation > 3
      augment_with_gap_safe <- ablation > 4

      printf(
        "\r%s, it: %02d/%02d, ablation level: %s",
        format(Sys.time(), "%H:%M:%S"),
        i,
        max_it,
        ablation
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
        augment_with_gap_safe = augment_with_gap_safe,
        hessian_warm_starts = hessian_warm_starts,
        update_hessian = update_hessian,
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
        n = n,
        p = p,
        ablation = ablation,
        rho = rho,
        it = i,
        time = fit$full_time,
        converged = all(fit$converged)
      )

      out <- rbind(out, res)
    }
  }
}

cat("\n")

saveRDS(out, "results/ablation.rds")
