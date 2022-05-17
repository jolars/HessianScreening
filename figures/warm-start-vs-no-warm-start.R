library(tibble)
library(tidyr)
library(dplyr)
library(tikzDevice)
library(ggplot2)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 5.6
fig_height <- 2

conf_level <- 0.05

d_raw <- readRDS("results/warm-start-vs-no-warm-start.rds") %>%
  filter(
    converged == TRUE
  ) %>%
  mutate(
    method = recode(
      method,
      "hessian_warmstart" = "Hessian (with warm starts)",
      "hessian" = "Hessian (without warm starts)",
      "working" = "Working"
    )
  ) %>%
  select(rho, method, time)

d1 <-
  d_raw %>%
  mutate(
    method = as.factor(method),
    family = as.factor(family),
    rho = as.factor(rho),
  ) %>%
  group_by(rho, method) %>% 
  summarize(
    meantime = mean(time),
    se = sd(time) / sqrt(n()),
    ci = qnorm(1 - conf_level/2) * se
  ) %>%
  mutate(
    hi = meantime + ci,
    lo = meantime - ci,
    rel_time = meantime / min(meantime),
    hi = hi / min(meantime),
    lo = lo / min(meantime)
  ) %>%
  drop_na(meantime)

cols <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

# file <- "figures/warm-starts-vs-no-warm-starts.tex"
# tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
ggplot(d1, aes(
  rho,
  rel_time,
  fill = method
)) +
  geom_col(position = position_dodge(0.9), col = 1) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = position_dodge(0.9),
    width = 0.25
  ) +
  scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
  scale_fill_manual(
    values = cols[c(1, 6, 2)], guide = guide_legend(nrow = 2)
  ) +
  theme(legend.position = "bottom") +
  labs(
    fill = NULL,
    x = NULL,
    y = "Time (relative)"
  ) 
# dev.off()
# renderPdf(file)
