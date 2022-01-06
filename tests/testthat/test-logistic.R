test_that("test logistic regression on real data", {
  library(HessianScreening)

  datalist <- "leukemia"

  tol_gap <- 1e-3
  standardize <- FALSE

  for (dataset in datalist) {

    data(list = list(dataset))
    d <- get(dataset)
    x <- d$X
    y <- d$y

    for (screening_type in c("working", "hessian", "gap_safe", "celer")) {
      fit <- lassoPath(
        x,
        y,
        "binomial",
        screening_type = screening_type,
        standardize = standardize,
        tol_gap = tol_gap,
        celer_use_accel = FALSE,
        celer_use_old_dual = FALSE
      )

      gaps <- duality_gaps(
        fit,
        "binomial",
        standardize = standardize,
        x,
        y
      )$gaps

      expect_true(all(gaps <= tol_gap))
    }
  }
})
