# test_that("all screening rules work without errors", {
#   n <- 100
#   p <- 50

#   d <- generateDesign(n, p)

#   fit <- lassoPath(d$X, d$y, screening_type = "hessian_adaptive", verbosity = 1)
# })