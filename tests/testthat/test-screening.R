test_that("screening methods work", {
    n <- 100
    p <- 1000

    d <- generateDesign(n, p)

    X <- d$X
    y <- d$y

    fit_hess <- lassoPath(X, y, screening_type = "hessian")
    fit_work <- lassoPath(X, y, screening_type = "working")
    fit_gap <- lassoPath(X, y, screening_type = "gap-safe")

})
