library(tibble)
library(tidyr)
library(dplyr)
library(tikzDevice)
library(ggplot2)
library(scales)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 5.6
fig_height <- 2.2

conf_level <- 0.05

d_raw <- readRDS("results/stopping-threshold.rds") %>%
  filter(
    !(screening_type %in% c("strong", "edpp", "gap_safe")),
    converged == TRUE
  ) %>%
  mutate(
    screening_type = recode_methods(screening_type),
    family = recode(
      family,
      "gaussian" = "Least-Squares",
      "binomial" = "Logistic"
    )
  ) %>%
  select(tol_gap, family, screening_type, time)

d1 <-
  d_raw %>%
  mutate(
    screening_type = as.factor(screening_type),
    family = as.factor(family)
  ) %>%
  group_by(family, tol_gap, screening_type) %>% 
  summarize(
    meantime = mean(time),
    se = sd(time) / sqrt(n()),
    ci = qnorm(1 - conf_level/2) * se
  ) %>%
  mutate(
    hi = meantime + ci,
    lo = meantime - ci
  )

cols <- c(
  "black", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

file <- "figures/stopping-threshold.tex"
tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
ggplot(
  d1,
  aes(tol_gap, meantime, color = screening_type, fill = screening_type)
) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, color = "transparent") +
  geom_line(size = 1) +
  facet_wrap("family") +
  theme(legend.position = c(0.88, 0.82), legend.title = element_blank()) +
  scale_fill_manual(values = cols) +
  scale_color_manual(values = cols) +
  scale_y_log10() +
  scale_x_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  labs(
    fill = "Method",
    color = "Method",
    x = "Convergence Threshold",
    y = "Time (s)"
  ) 
dev.off()

renderPdf("figures/stopping-threshold.tex")
