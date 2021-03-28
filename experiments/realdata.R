library(HessianScreening)
library(tidyr)
library(tibble)
library(Matrix)

printf <- function(...) invisible(cat(sprintf(...)))

datasets <- c(
  "arcene",
  "abalone",
  "cadata",
  "colon-cancer",
  "leukemia-train",
  "e2006-tfidf-train",
  "ijcnn1-train",
  "YearPredictionMSD-train",
  "madelon-train",
  "e2006-log1p-test",
  "news20"
)

g <- expand_grid(
  dataset = datasets,
  screening_type = c("working", "hessian", "gap_safe", "edpp"),
  family = NA,
  n = NA,
  p = NA,
  density = NA,
  time = NA,
  total_violations = NA,
  avg_screened = NA,
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

  dens <- Matrix::nnzero(X) / (n * p)

  family <- if (length(unique(d$y)) == 2) "binomial" else "gaussian"

  if (family == "gaussian" && screening_type == "edpp") {
    next
  }

  printf("%02d/%i %-10.10s %s\n", i, nrow(g), g$dataset[i], screening_type)

  n_it <- 1

  time <- double(n_it)

  for (k in seq_len(n_it)) {
    fit <- lassoPath(
      X,
      y,
      family = family,
      screening_type = screening_type,
      verbosity = 0
    )

    time[k] <- fit$full_time
  }

  g$n[i] <- n
  g$p[i] <- p
  g$family[i] <- family
  g$time[i] <- mean(time)
  g$density[i] <- dens
  g$total_violations[i] <- sum(fit$violations)
  g$avg_screened[i] <- mean(fit$active / fit$screened)
  g$violations[i] <- list(fit$violations)
  g$screened[i] <- list(fit$screened)
  g$active[i] <- list(fit$active)
}

saveRDS(g, "results/realdata.rds")
