library(tibble)
library(tidyr)
library(dplyr)
library(stringr)
library(tikzDevice)
library(ggplot2)
library(patchwork)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 5.6
fig_height <- 4.1

d_raw <- readRDS("results/hessian-time-frac.rds") %>%
  as_tibble() %>%
  mutate(
    dataset = str_remove(dataset, "(-train|-test)"),
    active = as.integer(active),
    screening_type = recode(
      screening_type,
      "hessian" = "Hessian",
      "hessian_approx" = "Hessian (Approx)",
      "hessian_full" = "Hessian (Full)",
      "working" = "Working"
    )
  )

d_active <-
  d_raw %>%
  filter(screening_type == "Working")

d_time <- 
  d_raw %>%
  select(-active, -total_time) %>%
  pivot_longer(
    c("kkt_time", "cd_time", "hess_time", "gradcorr_time"),
    names_to = "source",
    values_to = "time"
  ) %>%
  mutate(
    source = str_remove(source, "_time"),
    source = recode(
      source,
      "kkt" = "KKT",
      "cd" = "CD",
      "hess" = "Hessian Updates",
      "gradcorr" = "Hessian Correlation Estimate"
    )
  )

d_rcv1_time <- filter(d_time, dataset == "rcv1")
d_madelon_time <- filter(d_time, dataset == "madelon")
d_tfidf_time <- filter(d_time, dataset == "e2006-tfidf")

d_rcv1_active <- filter(d_active, dataset == "rcv1")
d_madelon_active <- filter(d_active, dataset == "madelon")
d_tfidf_active <- filter(d_active, dataset == "e2006-tfidf")

cols <- color_palette("tableau")

plot_function <- function(time_data, active_data) {
  pl_time <- 
    ggplot(time_data, aes(step, time, fill = source)) +
    geom_col() +
    facet_grid(rows = vars(screening_type)) +
    scale_fill_manual(values = cols) +
    theme(legend.position = c(0.16, 0.85)) +
    labs(y = "Time (s)", x = NULL, fill = NULL)

  pl_active <- 
    ggplot(active_data, aes(step, active)) +
    geom_col(fill = "transparent") +
    geom_line() +
    labs(y = "Active Predictors", x = "Step")

  pl_time / pl_active + plot_layout(heights = c(3, 1))
}
# plot_function(d_rcv1_time, d_rcv1_active)
# plot_function(d_madelon_time, d_madelon_active)
# plot_function(d_tfidf_time, d_tfidf_active)

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

files <- paste0(
  "figures/hessian-time-frac-",
  c("rcv1", "madelon", "tfidf"),
  ".tex"
)

tikz(files[1], width = fig_width, height = fig_height, standAlone = TRUE)
plot_function(d_rcv1_time, d_rcv1_active)
dev.off()
renderPdf(files[1])

tikz(files[2], width = fig_width, height = fig_height, standAlone = TRUE)
plot_function(d_madelon_time, d_madelon_active)
dev.off()
renderPdf(files[2])

tikz(files[3], width = fig_width, height = fig_height, standAlone = TRUE)
plot_function(d_tfidf_time, d_tfidf_active)
dev.off()
renderPdf(files[3])

