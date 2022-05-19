library(HessianScreening)
library(readr)

set.seed(3)

density <- 1 
rho <- 0.4
verbosity <- 1
tol_gap <- 1e-4
maxit <- 1e6
standardize <- TRUE
path_length <- 100

set.seed(723)
# d <- generateDesign(200, 20000, family = family, rho = rho, density = density)
d <- readRDS("data/news20.rds")
X <- d$X
y <- d$y

n <- nrow(X)
p <- ncol(X)

dens <- ifelse(inherits(X, "sparseMatrix"), Matrix::nnzero(X) / length(X), 1)
sparsity <- 1 - dens

family <- if (length(unique(d$y)) == 2) "binomial" else "gaussian"

log_hessian_update_type <-
  ifelse(sparsity * n / max(n, p) < 0.001, "full", "approx")

n <- nrow(X)
p <- ncol(X)

set.seed(723)

fit_hessian <- lassoPath(
  X,
  y,
  family = family,
  screening_type = "hessian",
  standardize = standardize,
  path_length = path_length,
  verbosity = verbosity,
  tol_gap = tol_gap,
  gap_safe_active_start = TRUE,
  log_hessian_update_type = log_hessian_update_type,
  celer_use_accel = TRUE,
  celer_prune = TRUE,
  maxit = maxit,
  store_dual_variables = TRUE,
  check_frequency = 1
)

print(paste0("time:", fit_hessian$full_time))
