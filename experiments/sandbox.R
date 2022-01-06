library(HessianScreening)
library(glmnet)
library(testthat)

set.seed(3)

family <- "binomial"

# d <- readRDS(file.path("data", paste0("arcene", ".rds")))
set.seed(14)
d <- generateDesign(1000, 100, family = family)
X <- d$X
y <- d$y

if (family != "binomial") {
  y <- y - mean(y)
}

n <- nrow(X)
p <- ncol(X)
verbosity <- 2
line_search <- 3
tol_gap <- 1e-4
maxit <- 1e7
standardize <- FALSE

fit_celer <- lassoPath(
  X,
  y,
  family = family,
  screening_type = "working",
  standardize = standardize,
  verbosity = verbosity,
  tol_gap = tol_gap,
  line_search = line_search,
  # log_hessian_update_type = "full",
  gap_safe_active_start = TRUE,
  celer_use_accel = FALSE,
  maxit = maxit
)

real_gaps <- duality_gaps(fit_celer, family, standardize, X, y)$gaps

celer_gaps <- (fit_celer$primals - fit_celer$duals) / fit_celer$primals

print(cbind(real_gaps, celer_gaps))
