test_that("gaussian and logistic models for simulated data", {
  library(HessianScreening)

  grid <- expand.grid(
    np = list(c(100, 5), c(50, 200)),
    density = c(0.5, 1),
    screening_type = c(
      "working",
      "hessian",
      "celer",
      "gap_safe",
      "edpp",
      "blitz"
    ),
    family = c("gaussian", "binomial"),
    standardize = c(FALSE),
    stringsAsFactors = FALSE
  )

  tol_gap <- 1e-4

  for (i in seq_len(nrow(grid))) {
    set.seed(i)

    g <- grid[i, ]

    np <- g$np[[1]]
    n <- np[1]
    p <- np[2]
    family <- g$family
    standardize <- g$standardize
    screening_type <- g$screening_type
    density <- g$density

    if (screening_type == "edpp" && family == "binomial") {
      next
    }

    data <- generateDesign(
      n,
      p,
      family = g$family,
      density = g$density
    )

    X <- data$X
    y <- data$y

    if (family == "gaussian") {
      y <- y - mean(y)
    }

    fit <- lassoPath(
      X,
      y,
      family,
      verbosity = 0,
      screening_type = screening_type,
      standardize = standardize,
      tol_gap = tol_gap,
      celer_use_old_dual = FALSE,
      celer_use_accel = TRUE,
      celer_prune = FALSE,
      store_dual_variables = TRUE
    )

    gaps <- check_gaps(fit, standardize, X, y, tol_gap)

    expect_true(all(gaps$below_tol))
  }
})
