library(HessianScreening)
library(readr)

set.seed(3)

family <- "binomial"
density <- 0.3

set.seed(14)
d <- generateDesign(100, 3, family = family, density = density)
X <- d$X
y <- d$y

if (family != "binomial") {
  y <- y - mean(y)
}

n <- nrow(X)
p <- ncol(X)
verbosity <- 1
line_search <- 0
tol_gap <- 1e-4
maxit <- 1e4
standardize <- FALSE

fit_celer <- lassoPath(
  X,
  y,
  family = family,
  screening_type = "blitz",
  standardize = standardize,
  verbosity = verbosity,
  tol_gap = tol_gap,
  line_search = line_search,
  gap_safe_active_start = TRUE,
  celer_use_accel = FALSE,
  celer_prune = TRUE,
  maxit = maxit
)

lambda <- fit_celer$lambda

y_celer <- ifelse(y == 1, 1, -1)

real_gaps <- check_gaps(fit_celer, family, standardize, X, y, tol_gap)

# celer_gaps <- (fit_celer$primals - fit_celer$duals) / fit_celer$primals

# print(cbind(real_gaps$gaps, celer_gaps))

print(real_gaps)
