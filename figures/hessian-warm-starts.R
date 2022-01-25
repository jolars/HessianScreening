library(ggplot2)
library(tikzDevice)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

fig_width <- 6.8
fig_height <- 2

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

dat <- readRDS("results/warm-starts.rds")

file <- "figures/hessian-warm-starts.tex"
# tikz(file, width = fig_width, height = fig_height, standAlone = TRUE)
ggplot(dat, aes(Step, Passes, col = WarmStart)) +
  geom_step() +
  facet_wrap(~dataset, scales = "free") +
  labs(col = "Warm Start", linetype = "Warm Start") +
  scale_color_manual(values = c("dark orange", "black"))
# dev.off()

# renderPdf(file)
