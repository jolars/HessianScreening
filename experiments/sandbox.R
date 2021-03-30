library(HessianScreening)

d <- readRDS(file.path("data", paste0("gisette-train", ".rds")))
X <- d$X
y <- d$y
n <- nrow(X)
p <- ncol(X)
verbosity <- 2
line_search = 2
family <- "binomial"
screening_type <- "hessian"
tol_gap <- 1e-4
tol_infeas <- 1e-4

sparsity <- 1 - Matrix::nnzero(X) / length(X)
sparsity * min(n, p) / max(n, p) * 10

fit_hessian <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "hessian",
    verbosity = verbosity,
    tol_gap = tol_gap,
    tol_infeas = tol_infeas,
    log_hessian_update_type = "approx",
    log_hessian_auto_update_freq = 10,
    line_search = line_search
)
fit_working <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "working",
    verbosity = verbosity,
    tol_gap = tol_gap,
    tol_infeas = tol_infeas,
    line_search = line_search)


# # cat("***************\n")
 cat("hessian:\n")
 cat("full = ", fit_hessian$full_time, "\n")
 cat("passes = ", sum(fit_hessian$passes), "\n")
# cat("cd_time = ", sum(fit$cd_time), "\n")
# cat("kkt_time = ", sum(fit$kkt_time), "\n")
# cat("hess_time = ", sum(fit$hess_time), "\n")
# cat("gradcorr_time", sum(fit$gradcorr_time), "\n")
# cat("***************\n")
 cat("working:\n")
 cat("full = ", fit_working$full_time, "\n")
 cat("passes = ", sum(fit_working$passes), "\n")
# cat("cd_time = ", sum(fit.w$cd_time), "\n")
# cat("kkt_time = ", sum(fit.w$kkt_time), "\n")


# plot(fit$passes, type = "l")
# lines(fit.w$passes, col = "red")
