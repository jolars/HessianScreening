
library(ggplot2)
library(dplyr)
library(tidyr)

d <- readRDS("results/simulateddata.rds")

d2 <-
  d %>%
  drop_na() %>%
  unnest(c(screened, active)) %>%
  group_by(family, scenario, screening_type) %>%
  mutate(screened = screened / p, active = active / p) %>%
  mutate(step = seq_along(screened))

ggplot(d2, aes(step)) +
  facet_wrap(~dataset) +
  geom_line(aes(y = active), linetype = 2) +
  geom_line(aes(y = screened, col = screening_type)) +
  # coord_cartesian(clip = "off") +
  scale_y_continuous(
    oob = function(x, ...) {
      x
    },
    limits = c(0, 0.06)
  )
