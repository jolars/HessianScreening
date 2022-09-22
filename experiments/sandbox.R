library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

printf <- function(...) invisible(cat(sprintf(...)))

n <- 100
p <- 10000
snr <- 2
rho <- 0.8
s <- 20
tol_gap <- 1e-4
path_length <- 100
family <- "gaussian"
rhos <- c(0, 0.4, 0.8)
gammas <- exp(seq(log(0.001), log(0.1), length.out = 10))

d <- generateDesign(n, p, family = family, rho = rho, snr = snr)
d <- readRDS("data/e2006-tfidf-train.rds")
X <- d$X
y <- d$y

set.seed(723)
fit <- lassoPath(
  X,
  y,
  verbosity = 1
)

set.seed(723)
fit_b <- lassoPath(
  X,
  y,
  verbosity = 1,
  screening_type = "working"
)

fit_work <- lassoPath(
  X,
  y,
  screening_type = "working",
  log_hessian_update = "approx",
  verbosity = 1,
  family = "binomial"
)

cbind(fit$active, fit$screened, fit$violations)
