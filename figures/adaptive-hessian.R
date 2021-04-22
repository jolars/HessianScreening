library(ggplot2)
library(tikzDevice)
library(tibble)

theme_set(theme_minimal(base_size = 9))
j
options(
  tikzDocumentDeclaration =
    "\\documentclass[10pt]{article}\n\\usepackage{newtxtext,newtxmath}\n"
)

d <- readRDS("results/adaptive-hessian.rds")

mrk <- 16087 / 100
cols <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

hline_data <- tribble(
  ~dataset, ~m,
  "e2006-tfidf", 16087 / 100,
  "madelon", 500 / 100
)

tikz("figures/adaptive-vs-grid.tex", width = 5.6, height = 1.6, standAlone = TRUE)
ggplot(d, aes(step, newactive, col = method)) +
  geom_hline(aes(yintercept = m), data = hline_data, linetype = 3) +
  geom_line() +
  labs(x = "Step", y = "Activated Predictors") +
  theme(legend.title = element_blank()) +
  scale_color_manual(values = c(cols[2], 1)) +
  facet_wrap(~dataset, scales = "free_y")
dev.off()

renderPdf("figures/adaptive-vs-grid.tex")
