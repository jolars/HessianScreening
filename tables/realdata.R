library(tibble)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)

l <- list.files("results/realdata", full.names = TRUE)

r <- lapply(l, readRDS)

d_raw <- do.call(bind_rows, r)

conf_level <- 0.05

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
    dataset = recode(
      dataset,
      "bc_tcga" = "bcTCGA"
    ),
    model = recode(
      family,
      "gaussian" = "Least-Squares",
      "binomial" = "Logistic"
    ),
    dataset = str_remove(dataset, "(-train|-test)")
  )

d_summary <- 
  d %>%
  group_by(dataset, model, n, p, density, screening_type) %>%
  summarize(time = mean(time)) %>%
  arrange(model, dataset, screening_type) %>%
  pivot_wider(names_from = "screening_type", values_from = "time")

d_details <-
  d %>%
  group_by(dataset, model, n, p, density, screening_type) %>%
  summarize(
    mean_time = mean(time),
    se = sd(time)/sqrt(n()),
    ci = qnorm(1 - conf_level/2) * se,
    lo = mean_time - ci,
    hi = mean_time + ci
  ) %>%
  select(dataset, n, p, density, model, screening_type, mean_time, lo, hi)

write_csv(d_summary, "tables/realdata-timings.csv")
write_csv(d_details, "tables/realdata-timings-details.csv")

