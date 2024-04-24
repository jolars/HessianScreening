library(tibble)
library(tidyr)
library(dplyr)
library(tikzDevice)
library(ggplot2)
library(patchwork)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 5.6
fig_height <- 2.4

conf_level <- 0.05

d_raw <- readRDS("results/gamma.rds") %>%
  as_tibble() %>%
  filter(converged == TRUE)

d1 <-
  d_raw %>%
  mutate(
    rho = factor(rho, ordered = TRUE)
  ) %>%
  group_by(rho) %>%
  mutate(mean_group_time = mean(time)) %>%
  group_by(rho, gamma) %>% 
  summarize(
    mean_time = mean(time) / max(mean_group_time),
    abs_time = mean(time),
    mean_violations = mean(violations),
    mean_active = mean(active),
    mean_screened = mean(screened)
  )

cols <- c("#4e79a7", "#f28e2b", "#e15759")

p_violations <- 
  ggplot(d1, aes(gamma, mean_violations, color = rho)) +
    geom_vline(xintercept = 0.01, linetype = 3, color = "black") +
    geom_line() +
    scale_x_log10() +
    scale_color_manual(values = cols) +
    labs(color = "$\\rho$", y = "Violations", x = "$\\gamma$")

p_time <-
  ggplot(d1, aes(gamma, mean_time, color = rho)) +
    geom_vline(xintercept = 0.01, linetype = 3, color = "black") +
    geom_line() +
    scale_x_log10() +
    ylim(0, NA) +
    scale_color_manual(values = cols) +
    labs(color = "$\\rho$", y = "Time (relative)", x = "$\\gamma$")

p_screened <-
  ggplot(d1, aes(gamma, mean_screened, color = rho)) +
    geom_vline(xintercept = 0.01, linetype = 3, color = "black") +
    geom_line() +
    scale_x_log10() +
    scale_color_manual(values = cols) +
    labs(color = "$\\rho$", y = "Screened Predictors", x = "$\\gamma$")

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

file <- "figures/gamma.tex"
tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
p_screened + p_violations + p_time + plot_layout(guides = "collect")
dev.off()
renderPdf(file)
