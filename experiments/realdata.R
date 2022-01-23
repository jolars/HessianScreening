library(HessianScreening)
library(Matrix)

printf <- function(...) invisible(cat(sprintf(...)))

datasets <- c(
  # gaussian
  "e2006-tfidf-train",
  "e2006-log1p-train",
  "YearPredictionMSD-train",
  # binomial
  "arcene",
  "colon-cancer",
  "duke-breast-cancer",
  "ijcnn1-train",
  "madelon-train",
  "rcv1-train",
  "news20"
)

tol_gap <- 1e-4
screening_types <- c(
  "hessian",
  "working",
  # "edpp",
  # "gap_safe",
  "blitz",
  "celer"
)

path_length <- 100

n_sim <- length(datasets)
out <- data.frame()

it_sim <- 0

for (dataset in datasets) {
  it_sim <- it_sim + 1

  d <- readRDS(file.path("data", paste0(dataset, ".rds")))

  X <- d$X
  y <- d$y

  n <- nrow(X)
  p <- ncol(X)
  
  max_it <- if (dataset %in% c("colon-cancer", "duke-breast-cancer")) {
    20
  } else if (dataset %in% c(
    "ijcnn1-train",
    "arcene"
  )) {
    10
  } else {
    1
  }

  dens <- ifelse(inherits(X, "sparseMatrix"), Matrix::nnzero(X) / length(X), 1)
  sparsity <- 1 - dens

  family <- if (length(unique(d$y)) == 2) "binomial" else "gaussian"

  log_hessian_update_type <-
    ifelse(sparsity * n / max(n, p) < 0.001, "full", "approx")

  printf(
    "\r%02d/%i %-10s\n", it_sim, n_sim, dataset)

  for (i in 1:max_it) {
    set.seed(i)

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
      set.seed(i)

      if (family == "binomial" && screening_type == "edpp") {
        next
      }

      printf(
        "\r%s, it: %02d/%02d %-10s",
        format(Sys.time(), "%H:%M:%S"),
        i,
        max_it,
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
        dataset = dataset,
        n = n,
        p = p,
        family = family,
        density = dens,
        screening_type = screening_type,
        it = i,
        time = fit$full_time,
        converged = all(fit$converged)
      )

      out <- rbind(out, res)
    }
  }
}

saveRDS(out, "results/realdata.rds")
