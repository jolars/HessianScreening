library(tibble)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)

source("R/utils.R")

d_raw <- readRDS("results/efficiency-simulateddata.rds")

conf_level <- 0.05

d <-
  d_raw %>%
  filter(screening_type %in% c("edpp", "strong", "hessian")) %>%
  mutate(
    step = seq_along(screened),
    screening_type = recode_methods(screening_type),
    model = recode(
      family,
      "gaussian" = "Least-Squares",
      "binomial" = "Logistic"
    )
  ) %>%
  rename(method = screening_type) %>%
  group_by(model, rho, method) %>%
  summarize(screened = mean(screened), violations = mean(violations))

write_csv(d, "tables/violations.csv")

