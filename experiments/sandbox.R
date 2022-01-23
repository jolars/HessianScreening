library(HessianScreening)
library(readr)

set.seed(3)

family <- "gaussian"
density <- 1 

set.seed(14)
# d <- generateDesign(100, 1000, family = family, density = density)
d <- readRDS("data/e2006-tfidf-train.rds")
X <- d$X
y <- d$y

n <- nrow(X)
p <- ncol(X)
verbosity <- 1
tol_gap <- 1e-4
maxit <- 1e5
standardize <- TRUE

fit_hessian <- lassoPath(
  X,
  y,
  family = family,
  screening_type = "hessian",
  standardize = standardize,
  verbosity = verbosity,
  tol_gap = tol_gap,
  gap_safe_active_start = TRUE,
  celer_use_accel = TRUE,
  celer_prune = TRUE,
  maxit = maxit,
  store_dual_variables = TRUE,
  check_frequency = 1
)

lambda <- fit_hessian$lambda

fit_blitz <- lassoPath(
  X,
  y,
  family = family,
  lambda = lambda,
  screening_type = "blitz",
  standardize = standardize,
  verbosity = verbosity,
  tol_gap = tol_gap,
  gap_safe_active_start = TRUE,
  celer_use_accel = TRUE,
  celer_prune = FALSE,
  maxit = maxit,
  store_dual_variables = TRUE,
  check_frequency = 1
)


# fit_blitz <- lassoPath(
#   X,
#   y,
#   family = family,
#   screening_type = "blitz",
#   standardize = standardize,
#   verbosity = verbosity,
#   tol_gap = tol_gap,
#   gap_safe_active_start = TRUE,
#   celer_use_accel = TRUE,
#   celer_prune = TRUE,
#   maxit = maxit,
#   store_dual_variables = TRUE,
#   check_frequency = 1
# )

# real_gaps <- check_gaps(fit, standardize, X, y, tol_gap)

# print(real_gaps)
print(cbind(fit_hessian$active, fit_blitz$active))
