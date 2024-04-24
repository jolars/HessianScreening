test_that("test logistic regression on real data", {
  library(HessianScreening)

  datalist <- "leukemia"

  tol_gap <- 1e-4
  standardize <- FALSE
  screening_types <- c("working", "hessian", "celer", "gap_safe", "blitz")

  for (dataset in datalist) {

    data(list = list(dataset))
    d <- get(dataset)
    x <- d$X
    y <- d$y

    for (screening_type in screening_types) {

      fit <- lassoPath(
        x,
        y,
        "binomial",
        screening_type = screening_type,
        standardize = standardize,
        tol_gap = tol_gap,
        celer_use_accel = FALSE,
        celer_use_old_dual = FALSE,
        verbosity = 0,
        store_dual_variables = TRUE
      )

      gaps <- check_gaps(fit, standardize, x, y, tol_gap)

      expect_true(all(gaps$below_tol))
    }
  }
})
