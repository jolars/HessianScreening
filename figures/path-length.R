library(tibble)
library(tidyr)
library(dplyr)
library(tikzDevice)
library(ggplot2)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 6.8
fig_height <- 2.5

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
  "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

# file <- "figures/simulateddata-timings.tex"
# tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
ggplot(
  d1,
  aes(path_length, meantime, color = screening_type, fill = screening_type)
) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, color = "transparent") +
  geom_line() +
  facet_grid(vars(family, np)) +
  theme(legend.position = c(0.1, 0.93), legend.title = element_blank()) +
  scale_fill_manual(values = cols) +
  scale_color_manual(values = cols) +
  labs(
    fill = "Method",
    color = "Method",
    x = "Path Length",
    y = "Time (s)"
  ) 
# dev.off()

# renderPdf("figures/simulateddata-timings.tex")
