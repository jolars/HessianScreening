library(tibble)
library(tidyr)
library(dplyr)
library(tikzDevice)
library(ggplot2)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 5.6
fig_height <- 2.4

conf_level <- 0.05

d_raw <- readRDS("results/path-length.rds") %>%
  filter(
    !(screening_type %in% c("strong", "edpp", "gap_safe")),
    converged == TRUE
  ) %>%
  mutate(
    screening_type = recode_methods(screening_type),
    np = paste0("$n=", n, "$, $p=", p, "$"),
    np = reorder(np, p),
    family = recode(
      family,
      "gaussian" = "Least-Squares",
      "binomial" = "Logistic"
    )
  ) %>%
  select(np, n, p, path_length, family, screening_type, time) %>%
  unnest(time)

d1 <-
  d_raw %>%
  mutate(
    screening_type = as.factor(screening_type),
    family = as.factor(family)
  ) %>%
  group_by(np, path_length, family, screening_type) %>% 
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

file <- "figures/path-length.tex"
tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
ggplot(
  d1,
  aes(path_length, meantime, color = screening_type, fill = screening_type)
) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, color = "transparent") +
  geom_line(size = 1) +
  facet_wrap(vars(family, np), nrow = 1) +
  theme(legend.position = c(0.06, 0.75), legend.title = element_blank()) +
  scale_fill_manual(values = cols) +
  scale_color_manual(values = cols) +
  labs(
    fill = "Method",
    color = "Method",
    x = "Path Length",
    y = "Time (s)"
  ) 
dev.off()

renderPdf("figures/path-length.tex")
