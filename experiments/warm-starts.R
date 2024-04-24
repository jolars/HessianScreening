library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

d <- readRDS("data/YearPredictionMSD-train.rds")
X <- d$X
y <- d$y

fit_warm1 <- lassoPath(X, y,
  hessian_warm_starts = TRUE,
  screening_type = "hessian",
  check_frequency = 1
)
fit_std1 <- lassoPath(X, y,
  hessian_warm_starts = FALSE,
  screening_type = "hessian",
  check_frequency = 1
)

d <- readRDS("data/colon-cancer.rds")
X <- d$X
y <- d$y

fit_warm2 <- lassoPath(X, y,
  family = "binomial",
  hessian_warm_starts = TRUE,
  screening_type = "hessian",
  check_frequency = 1
)
fit_std2 <- lassoPath(X, y,
  family = "binomial",
  hessian_warm_starts = FALSE,
  screening_type = "hessian",
  check_frequency = 1
)

n1 <- length(fit_warm1$lambda)
n2 <- length(fit_warm2$lambda)

dat <- tibble(
  dataset = rep(c("YearPredictionMSD", "colon-cancer"), times = c(n1, n2)),
  Step = c(1:n1, 1:n2),
  Hessian = c(fit_warm1$passes, fit_warm2$passes),
  Standard = c(fit_std1$passes, fit_std2$passes)
) %>%
  pivot_longer(c("Hessian", "Standard"),
    names_to = "WarmStart",
    values_to = "Passes"
  ) %>%
  mutate(Passes = as.integer(Passes))

saveRDS(dat, "results/warm-starts.rds")
