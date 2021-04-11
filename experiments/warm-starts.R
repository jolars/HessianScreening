
library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)

theme_set(theme_minimal(base_size = 9))

d <- readRDS("data/ijcnn1-train.rds")
X <- d$X
y <- d$y

fit_warm1 <- lassoPath(X, y,
  hessian_warm_starts = TRUE,
  screening_type = "hessian"
)
fit_std1 <- lassoPath(X, y,
  hessian_warm_starts = FALSE,
  screening_type = "hessian"
)

d <- readRDS("data/duke-breast-cancer.rds")
X <- d$X
y <- d$y

fit_warm2 <- lassoPath(X, y,
  family = "binomial",
  hessian_warm_starts = TRUE,
  screening_type = "hessian"
)
fit_std2 <- lassoPath(X, y,
  family = "binomial",
  hessian_warm_starts = FALSE,
  screening_type = "hessian"
)

n1 <- length(fit_warm1$lambda)
n2 <- length(fit_warm2$lambda)

dat <- tibble(
  dataset = rep(c("ijcnn1", "duke-breast-cancer"), times = c(n1, n2)),
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
