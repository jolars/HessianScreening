library(HessianScreening)

printf <- function(...) invisible(cat(sprintf(...)))

# xy <- readRDS("data/e2006-log1p-train.rds")
# xy <- readRDS("data/e2006-tfidf-train.rds")
# xy <- readRDS("data/rcv1-train.rds")
# xy <- readRDS("data/arcene.rds")
# xy <- readRDS("data/news20.rds")

# X <- xy$X
# y <- xy$y

# n <- nrow(X)
# p <- ncol(X)

tol_gap <- 1e-4
methods <- c("ours", "vanilla")

datasets <- c(
  "e2006-tfidf-train",
  "rcv1-train"
  # "ijcnn1-train",
  # "madelon-train",
  # "news20",
)

path_length <- 100

n_sim <- length(datasets)
out <- data.frame()

it_sim <- 0

# rho <- rhos

for (dataset in datasets) {
  it_sim <- it_sim + 1

  d <- readRDS(file.path("data", paste0(dataset, ".rds")))

  X <- d$X
  y <- d$y

  n <- nrow(X)
  p <- ncol(X)
  
  max_it <- if (dataset %in% c(
    "colon-cancer",
    "duke-breast-cancer",
    "arcene",
    "ijcnn1-train",
    "bc_tcga",
    "scheetz"
  )) {
    1
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

  set.seed(723)

  lambda <- fit$lambda

  for (it in 1:max_it) {
    for (method in methods) {
      if (method == "ours") {
        screening_type <- "hessian"
        update_hessian <- TRUE
      } else {
        screening_type <- "working"
        update_hessian <- FALSE
      }

      augment_with_gap_safe <- FALSE
      hessian_warm_starts <- TRUE

      printf(
        "\r%s, it: %02d/%02d, method: %s",
        format(Sys.time(), "%H:%M:%S"),
        it,
        max_it,
        method
      )
      flush.console()

      set.seed(723)

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
        verbosity = 1,
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
        method = method,
        it = it_sim,
        time = fit$full_time,
        converged = all(fit$converged)
      )

      out <- rbind(out, res)
    }

  }

}


cat("\n")

library(dplyr)
library(tidyr)
library(knitr)

out %>%
  select(dataset, method, time) %>%
  pivot_wider(names_from = "method", values_from = "time") %>%
  kable("simple", digits = 2)

