library(HessianScreening)
library(tidyr)
library(tibble)
library(Matrix)

printf <- function(...) invisible(cat(sprintf(...)))

datasets <- c(
  # gaussian
  # "cadata",
  # "e2006-tfidf-train",
  # "e2006-log1p-train",
  # "YearPredictionMSD-train",
  # binomial
  "arcene",
  "colon-cancer",
  "duke-breast-cancer",
  # "ijcnn1-train",
  # "madelon-train",
  # "rcv1-train",
  # "news20"
)

g <- expand_grid(
  dataset = datasets,
  screening_type = c(
    "working",
    "hessian",
    "gap_safe",
    "edpp",
    "celer",
    "blitz"
  ),
  family = NA,
  n = NA,
  p = NA,
  violations = list(NA),
  screened = list(NA),
  active = list(NA)
)

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

  printf("%02d/%i %-10.10s %s\n", i, nrow(g), g$dataset[i], screening_type)

  fit <- lassoPath(
    X,
    y,
    family = family,
    screening_type = screening_type,
    verbosity = 0,
    gap_safe_active_start = FALSE,
    log_hessian_update_type = log_hessian_update_type,
    line_search = 0
  )

  g$n[i] <- n
  g$p[i] <- p
  g$family[i] <- family
  g$violations[i] <- list(fit$violations)
  g$screened[i] <- list(fit$screened)
  g$active[i] <- list(fit$active)
}

cat("DONE!\n")

saveRDS(g, "results/efficiency-realdata.rds")
