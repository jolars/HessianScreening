
library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

d <- readRDS("data/e2006-tfidf-train.rds")

X <- d$X
y <- d$y

fit_adaptive_gaussian <-
  lassoPath(X, y,
    screening_type = "hessian_adaptive"
  )
fit_grid_gaussian <-
  lassoPath(X, y, screening_type = "hessian")

d <- readRDS("data/madelon-train.rds")

X <- d$X
y <- d$y

fit_adaptive_binomial <-
  lassoPath(X, y,
    family = "binomial",
    screening_type = "hessian_adaptive"
  )
fit_grid_binomial <-
  lassoPath(X, y, family = "binomial", screening_type = "hessian")

tmp <- tibble(
  step = c(
    seq_along(fit_adaptive_gaussian$lambda),
    seq_along(fit_grid_gaussian$lambda),
    seq_along(fit_adaptive_binomial$lambda),
    seq_along(fit_grid_binomial$lambda)
  ),
  newactive = c(
    fit_adaptive_gaussian$new_active,
    fit_grid_gaussian$new_active,
    fit_adaptive_binomial$new_active,
    fit_grid_binomial$new_active
  ),
  dataset = rep(
    c("e2006-tfidf", "madelon"),
    times = c(
      length(fit_adaptive_gaussian$lambda) +
        length(fit_grid_gaussian$lambda),
      length(fit_adaptive_binomial$lambda) +
        length(fit_grid_binomial$lambda)
    )
  ),
  method = rep(
    c("Adaptive", "Grid", "Adaptive", "Grid"),
    times = c(
      length(fit_adaptive_gaussian$lambda),
      length(fit_grid_gaussian$lambda),
      length(fit_adaptive_binomial$lambda),
      length(fit_grid_binomial$lambda)
    )
  )
) %>%
  group_by(dataset, method) %>%
  mutate(frac = step / max(step))

saveRDS(tmp, "results/adaptive-hessian.rds")
