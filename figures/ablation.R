library(tibble)
library(tidyr)
library(dplyr)
library(tikzDevice)
library(ggplot2)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 5
fig_height <- 2

conf_level <- 0.05

d_raw <- readRDS("results/ablation.rds") %>%
  as_tibble() %>%
  filter(
    converged == TRUE
  ) %>%
  mutate(
    ablation = recode_factor(
      ablation,
      "1" = "Vanilla",
      "2" = "Hessian Screening",
      "3" = "Hessian Warm Starts",
      "4" = "Hessian Updates",
      "5" = "Gap Safe",
      .ordered = TRUE
    ),
    rho = factor(rho, ordered = TRUE)
  ) %>%
  select(ablation, rho, time)

d <-
  d_raw %>%
  mutate(
    rho = as.factor(rho)
  ) %>%
  group_by(ablation, rho) %>% 
  summarize(
    meantime = mean(time),
    se = sd(time) / sqrt(n()),
    ci = qnorm(1 - conf_level/2) * se
  ) %>%
  mutate(
    hi = meantime + ci,
    lo = meantime - ci
  ) 

# cols <- c(
#   "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
#   "#0072B2", "#D55E00", "#CC79A7"
# )

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

file <- "figures/ablation.tex"
tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
ggplot(d, aes(rho, meantime, fill = ablation)) +
  geom_col(position = position_dodge(0.9), col = 1) +
  geom_errorbar(
    aes(ymin = lo, ymax = hi),
    position = position_dodge(0.9),
    width = 0.25
  ) +
  # theme(legend.position = c(0.6, 0.8), panel.grid.major.x = element_blank()) +
  labs(
    fill = NULL,
    x = "$\\rho$",
    y = "Time (s)"
  )  
dev.off()
renderPdf(file)
