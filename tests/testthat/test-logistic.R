test_that("test logistic regression on real data", {
  suppressMessages(library(glmnet))

  datalist <- c("heart", "leukemia")

  for (dataset in datalist) {

    data(list = list(dataset))
    d <- get(dataset)
    X <- d$X
    y <- d$y

    for (screening_type in c("working", "hessian", "gap_safe")) {

      fit_ours <- lassoPath(X,
        y,
        "binomial",
        screening_type = "hessian",
        standardize = FALSE
      )

      lambda <- fit_ours$lambda / nrow(X)

      fit_gnet <- glmnet(X,
        y,
        "binomial",
        lambda = lambda,
        intercept = FALSE,
        standardize = FALSE
      )

      ours_dev <- fit_ours$dev_ratio
      gnet_dev <- fit_gnet$dev.ratio

      expect_equal(ours_dev, gnet_dev, tolerance = 1e-3)

    }
  }
})
