library(HessianScreening)
library(tibble)
library(dplyr)
library(tidyr)
library(readr)
library(forcats)

printf <- function(...) invisible(cat(sprintf(...)))

g <- expand_grid(
  family = c("gaussian"),
  scenario = c(1, 2),
  n = NA,
  p = NA,
  rho = c(0, 0.4),
  screening_type = c("hessian", "working", "celer"),
  path_length = 100,
  avg_screened = NA,
  avg_violations = NA,
  time = list(NA),
  screened = list(NA),
  active = list(NA),
  step = list(NA)
)

n_it <- 5

for (i in seq_len(nrow(g))) {
  rho <- g$rho[i]
  family <- g$family[i]
  screening_type <- g$screening_type[i]
  path_length <- g$path_length[i]
  scenario <- g$scenario[i]

  if (family == "binomial" && screening_type == "edpp") {
    next
  }

  if (scenario == 1) {
    n <- 10000
    p <- 100
    snr <- 1
    s <- 5
  } else if (scenario == 2) {
    n <- 400
    p <- 40000
    snr <- 2
    s <- 20
  }

  avg_screened <- violations <- time <- double(n_it)
  active <- screened <- matrix(NA, nrow = n_it, ncol = path_length)

  printf(
    "%02d/%i %-10s n: %4d p: %4d rho: %1.1f %-10s\n",
    i, nrow(g), family, n, p, rho, screening_type
  )

  for (j in seq_len(n_it)) {
    set.seed(j)

    d <- generateDesign(n, p, family = family, rho = rho, snr = snr)
    X <- d$X
    y <- d$y

    fit <- lassoPath(
      X,
      y,
      family = family,
      screening_type = screening_type,
      path_length = path_length,
      log_hessian_update_type = "full",
      gamma = 0.01,
      tol_gap = 1e-6,
      tol_infeas = 1e-5,
      line_search = 3
    )

    n_lambda <- length(fit$lambda)

    time[j] <- fit$full_time
    avg_screened[j] <- mean(fit$active / fit$screened)
    violations[j] <- sum(fit$violations)
    screened[j, 1:n_lambda] <- fit$screened
    active[j, 1:n_lambda] <- fit$active

    # # stop if standard error is within 2.5% of mean
    # if (j > 19) {
    #   time_se <- sd(time[1:j]) / sqrt(j)

    #   if (time_se / mean(time[1:j]) < 0.025) {
    #     break
    #   }
    # }
  }

  time <- time[1:j]
  active <- active[1:j, ]
  screened <- screened[1:j, ]
  violations <- violations[1:j]
  avg_screened <- avg_screened[1:j]

  dontuse <- apply(screened, 2, anyNA)

  active <- colMeans(active)
  screened <- colMeans(screened)

  g$n[i] <- n
  g$p[i] <- p
  g$family[i] <- family
  g$time[i] <- list(time)
  g$avg_screened[i] <- mean(avg_screened)
  g$avg_violations[i] <- mean(violations)
  g$screened[i] <- list(screened)
  g$active[i] <- list(active)
  g$step[i] <- list(1:path_length)
}

write_rds(g, "results/response-sims.rds")

source("R/utils.R")

d_raw <- read_rds("results/response-sims.rds") %>%
  mutate(
    screening_type = recode(
      screening_type,
      "hessian" = "Hessian",
      "working" = "Working",
      "gap_safe" = "Gap Safe",
      "edpp" = "EDPP",
      "celer" = "Celer"
    ),
    rho = as.factor(rho),
    np = paste0("$n=", n, "$, $p=", p, "$"),
    np = reorder(np, p),
  ) %>%
  select(np, n, p, rho, family, screening_type, time) %>%
  unnest(time)

d1 <-
  d_raw %>%
  mutate(
    screening_type = as.factor(screening_type),
    family = as.factor(family)
  ) %>%
  group_by(np, rho, screening_type) %>%
  summarize(meantime = mean(time))


d1 %>%
  select(rho, screening_type, meantime) %>%
  pivot_wider(names_from = "screening_type",
              values_from = "meantime") %>%
  rename("$\\rho$" = "rho", "scenario" = "np") %>%
  kableExtra::kbl(format = "pipe", digits = 2)

