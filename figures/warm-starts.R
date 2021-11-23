
library(ggplot2)
library(tikzDevice)

source("R/utils.R")

theme_set(theme_minimal(base_size = 9))

options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

dat <- readRDS("results/warm-starts.rds")

pl <- 
  ggplot(dat, aes(Step, Passes, col = WarmStart)) +
    geom_step() +
    facet_wrap(~dataset, scales = "free") +
    labs(col = "Warm Start", linetype = "Warm Start") +
    scale_color_manual(values = c("dark orange", "black"))

file <- "figures/hessian-warm-starts.tex"
tikz(file, width = 5.6, height = 2, standAlone = TRUE)
print(pl)
dev.off()

renderPdf(file)

