library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

printf <- function(...) invisible(cat(sprintf(...)))

n <- 10000
p <- 100
snr <- 2
rho <- 0.0
s <- 20
tol_gap <- 1e-4
path_length <- 100
family <- "poisson"
rhos <- c(0)
verbosity <- 1

d <- generateDesign(n, p, family = family, rho = rho, snr = snr)
X <- d$X
y <- d$y

# set.seed(2)
fit_a <- lassoPath(
  X,
  y,
  verbosity = verbosity,
  screening_type = "blitz",
  family = family,
  line_search = TRUE
)

# set.seed(723)
# fit_b <- lassoPath(
#   X,
#   y,
#   verbosity = 1,
#   screening_type = "working",
#   family = family
# )

# fit_a$full_time
# fit_b$full_time
