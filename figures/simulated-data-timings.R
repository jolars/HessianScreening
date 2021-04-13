library(tibble)
library(tidyr)
library(dplyr)
library(forcats)
library(lattice)
library(latticeExtra)
library(tactile)
library(tikzDevice)
library(ggplot2)

theme_set(theme_minimal(base_size = 9))

d_raw <- readRDS("results/simulateddata.rds") %>%
  mutate(
    screening_type = recode(
      screening_type,
      "hessian" = "Hessian",
      "working" = "Working",
      "gap_safe" = "Gap-Safe",
      "edpp" = "EDPP"
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
  group_by(np, rho, family, screening_type) %>%
  summarize(
    meantime = mean(time),
    se = sd(time / n())
  ) %>%
  mutate(
    hi = meantime + se,
    lo = meantime - se,
    rel_time = meantime / min(meantime)
  )

d2_gaussian <-
  filter(d1, family == "gaussian")

d2_binomial <-
  filter(d1, family == "binomial") %>%
  drop_na(meantime)

cols <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

library(ggthemes)

tikz("figures/simulateddata-gaussian-timings.tex", width = 4.8, height = 2.5)
ggplot(d2_gaussian, aes(
  rho,
  rel_time,
  fill = screening_type
)) +
  geom_col(position = "dodge", col = 1) +
  facet_wrap("np") +
  scale_fill_manual(values = cols[1:4]) +
  labs(
    fill = "Screening",
    x = "Correlation ($\\rho$)",
    y = "Time"
  )
dev.off()

tikz("figures/simulateddata-binomial-timings.tex", width = 4.8, height = 2.5)
ggplot(d2_binomial, aes(
  rho,
  rel_time,
  fill = screening_type
)) +
  geom_col(position = "dodge", col = 1) +
  facet_wrap("np") +
  scale_fill_manual(values = cols[2:4]) +
  labs(
    fill = "Screening",
    x = "Correlation ($\\rho$)",
    y = "Time"
  )
dev.off()
