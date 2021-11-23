library(HessianScreening)
library(readr)

n <- 200
p <- 20000
family <- "gaussian"

rho <- 0.5

for (i in 1:20) {
  set.seed(i)
  d <- generateDesign(n, p, family = family, rho = rho, snr = 0.1)

  X <- d$X
  y <- d$y

  file <- paste0("data/simulated_data/", i, ".rds")

  write_rds(list(X = X, y = y), file)
}
