
library(ggplot2)
library(tikzDevice)

dat <- readRDS("results/warm-starts.rds")

cols <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
  "#0072B2", "#D55E00", "#CC79A7"
)

tikz("figures/hessian-warm-starts.tex", width = 4.5, height = 2)
ggplot(dat, aes(Step, Passes, col = WarmStart, linetype = WarmStart)) +
  geom_step() +
  facet_wrap(~dataset, scales = "free") +
  labs(col = "Warm Start", lty = "Warm Start") +
  scale_color_manual(values = cols[1:2])
dev.off()
