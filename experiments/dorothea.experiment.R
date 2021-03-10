rm(list = ls())
graphics.off()
library(HessianScreening)

d <- readRDS(file.path("data", paste0("dorothea", ".rds")))
X <- d$X
y <- d$y
family <- "binomial"
screening_type <- "hessian"
fit.w <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "working",
    verbosity = 1
)
fit <- lassoPath(
    X,
    y,
    family = family,
    screening_type = screening_type,
    hessian_warm_starts = TRUE,
    approx_hessian = F,
    verbosity = 1
)

# cat("***************\n")
cat("hessian:\n")
cat("full = ", fit$full_time, "\n")
cat("cd_time = ", sum(fit$cd_time), "\n")
cat("corr_time = ", sum(fit$corr_time), "\n")
cat("hess_time = ", sum(fit$hess_time), "\n")
cat("gradcorr_time", sum(fit$gradcorr_time), "\n")
cat("***************\n")
cat("working:\n")
cat("full = ", fit.w$full_time, "\n")
cat("cd_time = ", sum(fit.w$cd_time), "\n")
cat("corr_time = ", sum(fit.w$corr_time), "\n")


plot(fit$passes, type = "l")
lines(fit.w$passes, col = "red")
