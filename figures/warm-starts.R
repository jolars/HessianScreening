
library(ggplot2)
library(tikzDevice)

theme_set(theme_minimal(base_size = 9))

dat <- readRDS("results/warm-starts.rds")

tikz("figures/hessian-warm-starts.tex", width = 5.6, height = 2)
ggplot(dat, aes(Step, Passes, col = WarmStart)) +
  geom_step() +
  facet_wrap(~dataset, scales = "free") +
  labs(col = "Warm Start", linetype = "Warm Start") +
  scale_color_manual(values = c("dark orange", "black"))
dev.off()
