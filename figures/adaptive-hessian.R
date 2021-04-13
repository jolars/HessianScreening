library(ggplot2)
library(tikzDevice)

theme_set(theme_minimal(base_size = 9))

d <- readRDS("results/adaptive-hessian.rds")

mrk <- 16087 / 100
cols <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

tikz("figures/tfidf-adaptive-vs-grid.tex", width = 3.7, height = 2)
ggplot(d, aes(step, newactive, col = method)) +
  geom_hline(yintercept = mrk, linetype = 3) +
  geom_line() +
  labs(x = "Step", y = "Activated Predictors") +
  theme(legend.title = element_blank()) +
  scale_color_manual(values = cols[2:3])
dev.off()
