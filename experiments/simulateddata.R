library(HessianScreening)
library(RColorBrewer)

n <- 200
p <- 5000

d <- generateDesign(n, p)

X <- d$X
y <- d$y

fit_hess <- lassoPath(X, y, screening_type = "hessian")
#fit_work <- lassoPath(X, y, screening_type = "working")
fit_strong <- lassoPath(X, y, screening_type = "strong")
fit_gap <- lassoPath(X, y, screening_type = "gap_safe")
fit_edpp <- lassoPath(X, y, screening_type = "edpp", force_kkt_check = TRUE)

ylim <- extendrange(c(0, max(c(fit_hess$screened, fit_gap$screened))))
cols <- c(1, brewer.pal(8, "Dark2"))
lty <- c(2, 1, 1, 1, 1)

#library(tikzDevice)
#tikz("simulations.tex", width = 5, height = 5)
plot(fit_hess$active, ylim = ylim, type = "l", lty = 2,
     xlab = "Step",
     ylab = "Screened Predictors")

lines(fit_hess$screened, col = cols[2])
lines(fit_strong$screened, col = cols[3])
lines(fit_gap$screened, col = cols[4])
lines(fit_edpp$screened, col = cols[5])

labels <- c("active", "hessian", "strong", "Gap-SAFE", "EDPP")

legend("topleft", legend = labels,
       col = cols[seq_along(labels)], lty = lty)
#dev.off()
