library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

printf <- function(...) invisible(cat(sprintf(...)))

tol_gap <- 1e-4

path_length <- 100

out <- data.frame()

datasets <- c(
  # "e2006-log1p-train",
  "e2006-tfidf-train",
  "madelon-train",
  "rcv1-train"
)

screening_types <- c(
  "hessian_approx", "hessian_full", "working"
)

for (dataset in datasets) {
  d <- readRDS(file.path("data", paste0(dataset, ".rds")))

  X <- d$X
  y <- d$y

  n <- nrow(X)
  p <- ncol(X)
  
  dens <- ifelse(inherits(X, "sparseMatrix"), Matrix::nnzero(X) / length(X), 1)
  sparsity <- 1 - dens

  family <- if (length(unique(d$y)) == 2) "binomial" else "gaussian"

  log_hessian_update_type <-
    ifelse(sparsity * n / max(n, p) < 0.001, "full", "approx")

  printf("\r%s\n", dataset)

  set.seed(723)

  fit <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "hessian",
    path_length = path_length,
    log_hessian_update_type = ifelse(sparsity * n / max(n, p) < 0.001, "full", "approx"),
    verbosity = 0,
    tol_gap = tol_gap
  )

  lambda <- fit$lambda

  for (screening_type in screening_types) {
    set.seed(723)

    if (screening_type != "working") {
      if (family == "binomial") {
        log_hessian_update_type <- 
          if (screening_type == "hessian_full") "full" else "approx"
        screening_type_out <- screening_type
      } else {
        screening_type_out <- "hessian"
      }
      screening_type_in <- "hessian"
    } else {
      screening_type_in <- "working"
      screening_type_out <- "working"
    }

    printf("\r%s, %-10s\n", format(Sys.time(), "%H:%M:%S"), screening_type_out)

    fit <- lassoPath(
      X,
      y,
      family = family,
      lambda = lambda,
      screening_type = screening_type_in,
      path_length = path_length,
      log_hessian_update_type = log_hessian_update_type,
      celer_prune = TRUE,
      verbosity = 0,
      tol_gap = tol_gap
    )

    if (any(!fit$converged)) {
      warning("failed to converge")
    }

    if (screening_type_in == "hessian") {
      hess_time <- fit$hess_time
      gradcorr_time <- fit$gradcorr_time
      if (length(hess_time) != length(gradcorr_time)) {
        gradcorr_time <- c(gradcorr_time, 0)
      }
    } else {
      gradcorr_time <- hess_time <- double(length(fit$active))
    }

    stopifnot(length(hess_time) == length(fit$active))

    res <- data.frame(
      dataset = dataset,
      n = n,
      p = p,
      family = family,
      density = dens,
      screening_type = screening_type_out,
      step = seq_along(fit$active),
      active = fit$active,
      total_time = fit$it_time,
      kkt_time = fit$kkt_time,
      cd_time = fit$cd_time,
      hess_time = hess_time,
      gradcorr_time = gradcorr_time
    )

    out <- rbind(out, res)
  }
}

cat("\n")

saveRDS(out, "results/hessian-time-frac.rds")
