library(ggplot2)
library(dplyr)
library(tikzDevice)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 5.6
fig_height <- 1.4

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

dat <- readRDS("results/warm-starts.rds")

file <- "figures/hessian-warm-starts.tex"
tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
ggplot(dat, aes(Step, Passes, col = WarmStart)) +
  geom_step() +
  labs(col = NULL, linetype = "Warm Start") +
  scale_color_manual(values = c("dark orange", "black")) +
  facet_wrap("dataset", scales = "free_x") +
  theme(legend.position = c(0.07, 0.93))
dev.off()
renderPdf(file)
