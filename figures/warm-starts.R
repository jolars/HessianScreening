
library(ggplot2)
library(tikzDevice)

theme_set(theme_minimal(base_size = 9))

dat <- readRDS("results/warm-starts.rds")

tikz("figures/hessian-warm-starts.tex", width = 4.7, height = 2)
ggplot(dat, aes(Step, Passes, col = WarmStart, linetype = WarmStart)) +
  geom_step() +
  facet_wrap(~dataset, scales = "free") +
  labs(col = "Warm Start", linetype = "Warm Start") +
  scale_color_manual(values = c("dark grey", "dark orange"))
dev.off()
