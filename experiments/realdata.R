library(HessianScreening)
library(tidyr)
library(tibble)
library(Matrix)

printf <- function(...) invisible(cat(sprintf(...)))

datasets <- c(
  # gaussian
  "e2006-tfidf-train",
  "e2006-log1p-train",
  "YearPredictionMSD-train",
  # # binomial
  "arcene",
  "colon-cancer",
  "duke-breast-cancer",
  "ijcnn1-train",
  "madelon-train",
  "rcv1-train",
  "news20"
)

g <- expand_grid(
  dataset = datasets,
  screening_type = c(
    "working",
    "hessian",
    # "gap_safe",
    # "edpp",
    "blitz",
    "celer"
  ),
  family = NA,
  n = NA,
  p = NA,
  density = NA,
  time = NA,
  total_violations = NA,
  avg_screened = NA,
  violations = list(NA),
  screened = list(NA),
  active = list(NA),
  converged = NA
)

tol_gap <- 1e-4

min_it <- 2
max_it <- 1000
max_err <- 0.1
conf_level <- 0.05

for (i in seq_len(nrow(g))) {
  d <- readRDS(file.path("data", paste0(g$dataset[i], ".rds")))
  screening_type <- g$screening_type[i]

  X <- d$X
  y <- d$y

  n <- nrow(X)
  p <- ncol(X)

  dens <- ifelse(inherits(X, "sparseMatrix"), Matrix::nnzero(X) / length(X), 1)
  sparsity <- 1 - dens

  family <- if (length(unique(d$y)) == 2) "binomial" else "gaussian"

  if (family == "binomial" && screening_type == "edpp") {
    next
  }

  log_hessian_update_type <-
    ifelse(sparsity * n / max(n, p) < 0.001, "full", "approx")

  printf("\r%02d/%i %-10.10s %s\n", i, nrow(g), g$dataset[i], screening_type)

  time <- double(max_it)

  for (k in seq_len(max_it)) {
    set.seed(848)

    printf("\r%s, it: %02d", format(Sys.time(), "%H:%M:%S"), k)
    flush.console() 

    fit <- lassoPath(
      X,
      y,
      family = family,
      screening_type = screening_type,
      verbosity = 0,
      log_hessian_update_type = log_hessian_update_type,
      tol_gap = tol_gap
    )

    time[k] <- fit$full_time

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
    if (k > min_it) {
      se <- sd(time[1:k]) / sqrt(k)
      ci_width <- 2 * qnorm(1 - conf_level / 2) * se

      if (ci_width / mean(time[1:k]) < max_err) {
        break
      }
    }
  }

  g$n[i] <- n
  g$p[i] <- p
  g$family[i] <- family
  g$time[i] <- mean(time[1:k])
  g$density[i] <- dens
  g$total_violations[i] <- sum(fit$violations)
  g$avg_screened[i] <- mean(fit$active / fit$screened)
  g$violations[i] <- list(fit$violations)
  g$screened[i] <- list(fit$screened)
  g$active[i] <- list(fit$active)
  g$converged[i] <- all(fit$converged)
}

cat("DONE!\n")

saveRDS(g, "results/realdata.rds")
