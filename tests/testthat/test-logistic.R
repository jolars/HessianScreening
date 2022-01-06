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

    for (screening_type in c("working", "hessian", "celer", "gap_safe")) {
      fit <- lassoPath(
        x,
        y,
        "binomial",
        screening_type = screening_type,
        standardize = standardize,
        tol_gap = tol_gap,
        celer_use_accel = FALSE,
        celer_use_old_dual = FALSE,
        verbosity = 0
      )

      gaps <- check_gaps(fit, "binomial", standardize, x, y, tol_gap)

      expect_true(all(gaps$below_tol))
    }
  }
})
