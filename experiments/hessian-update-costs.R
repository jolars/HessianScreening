library(HessianScreening)
library(tibble)
library(dplyr)
library(ggplot2)
library(tidyr)

d <- readRDS("data/e2006-log1p-train.rds")
X <- d$X
y <- d$y

fit_hess <- lassoPath(X, y, screening_type = "hessian")
fit_work <- lassoPath(X, y, screening_type = "working", lambda = fit_hess$lambda)

steps <- seq_along(fit_hess$lambda)

d_hess <- tibble(
  step = seq_along(fit_hess$lambda),
  hess = fit_hess$hess_time,
  gradcorr = c(fit_hess$gradcorr_time, 0),
  kkt_time = fit_hess$kkt_time,
  cd = fit_hess$cd_time
) %>%
  pivot_longer(hess:cd, names_to = "component", values_to = "time")

d_working <- tibble(step = seq_along(fit_work$lambda), time = fit_work$it_time)

ggplot(d_hess, aes(step, time, fill = component)) +
  geom_area() +
  geom_line(data = d_working, aes(step, time, linetype = "working"), inherit.aes = FALSE)
