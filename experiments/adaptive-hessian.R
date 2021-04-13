
library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

d <- readRDS("data/e2006-tfidf-train.rds")

X <- d$X
y <- d$y

fit_adaptive <-
  lassoPath(X, y,
    screening_type = "hessian_adaptive",
    verbosity = 1
  )
fit_grid <-
  lassoPath(X, y, screening_type = "hessian")

tmp <- tibble(
  step = c(seq_along(fit_adaptive$lambda), seq_along(fit_grid$lambda)),
  newactive = c(fit_adaptive$new_active, fit_grid$new_active),
  method = rep(c("Adaptive", "Grid"), times = c(
    length(fit_adaptive$lambda),
    length(fit_grid$lambda)
  ))
) %>%
  group_by(method) %>%
  mutate(frac = step / max(step))

saveRDS(tmp, "results/adaptive-hessian.rds")
