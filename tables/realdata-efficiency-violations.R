library(tibble)
library(forcats)
library(readr)
library(tidyr)
library(dplyr)

d <- read_rds("results/realdata.rds")

d1 <-
  d %>%
  drop_na() %>%
  filter(screening_type != "working") %>%
  select(dataset, screening_type, total_violations, avg_screened)

print(d1, n = 100)
