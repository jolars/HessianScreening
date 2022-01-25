library(tibble)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)

d_raw <- readRDS("results/realdata.rds")

d <-
  as_tibble(d_raw) %>%
  drop_na(time) %>%
  mutate(
    screening_type = recode(
      screening_type,
      "working" = "Working",
      "hessian" = "Hessian",
      "gap_safe" = "GapSafe",
      "edpp" = "EDPP",
      "celer" = "Celer",
      "blitz" = "Blitz"
    ),
    family = recode(
      family,
      "gaussian" = "Least-Squares",
      "binomial" = "Logistic"
    ),
    dataset = str_remove(dataset, "(-train|-test)")
  ) %>%
  group_by(dataset, family, n, p, density, screening_type) %>%
  summarize(time = mean(time)) %>%
  arrange(family, dataset, screening_type) %>%
  pivot_wider(names_from = "screening_type", values_from = "time")

write_csv(d, "tables/realdata-timings.csv")

filter(d, family == "Gaussian") %>%
  select(-family) %>%
  write_csv("tables/realdata-timings-gaussian.csv")

filter(d, family == "Binomial") %>%
  select(-family) %>%
  write_csv("tables/realdata-timings-binomial.csv")
