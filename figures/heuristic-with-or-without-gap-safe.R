library(tibble)
library(tidyr)
library(dplyr)
library(tikzDevice)
library(ggplot2)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 3.35
fig_height <- 2.5

conf_level <- 0.05

d_raw <- readRDS("results/heuristic-with-or-without-gap-safe.rds") %>%
  filter(
    converged == TRUE
  ) %>%
  mutate(
    screening_type = recode(
      screening_type,
      "hessian" = "Hessian",
      "working" = "Working"
    )
  ) %>%
  select(screening_type, gap_safe, rho, time)

d <-
  d_raw %>%
  mutate(
    rho = as.factor(rho)
  ) %>%
  group_by(screening_type, gap_safe, rho) %>% 
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
  "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

file <- "figures/heuristic-with-or-without-gap-safe.tex"
tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
ggplot(d, aes(screening_type, meantime, fill = gap_safe)) +
  geom_col(position = position_dodge(0.9), col = 1) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = position_dodge(0.9),
    width = 0.25
  ) +
  scale_fill_manual(
    values = cols[c(1, 6)]  ) +
  theme(legend.position = c(0.2, 0.8), panel.grid.major.x = element_blank()) +
  labs(
    fill = NULL,
    x = NULL,
    y = "Time (s)"
  )  +
  facet_wrap(
    "rho",
    labeller = labeller(rho = rho_labeller)
  )
dev.off()
renderPdf(file)
