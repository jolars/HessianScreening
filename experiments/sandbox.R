library(HessianScreening)

d <- readRDS(file.path("data", paste0("dorothea", ".rds")))
X <- d$X
y <- d$y
n <- nrow(X)
p <- ncol(X)
verbosity <- 1
line_search <- 2
family <- "binomial"
screening_type <- "hessian"
tol_gap <- 1e-4
tol_infeas <- 1e-4

sparsity <- 1 - Matrix::nnzero(X) / length(X)
sparsity * min(n, p) / max(n, p)

n / max(n, p) * sparsity

fit_hessian <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "hessian",
    verbosity = verbosity,
    tol_gap = tol_gap,
    tol_infeas = tol_infeas,
    line_search = line_search,
    log_hessian_update_type = "full"
)
fit_working <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "working",
    verbosity = verbosity,
    tol_gap = tol_gap,
    tol_infeas = tol_infeas,
    line_search = line_search
)
if(family != "binomial"){
fit_edpp <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "edpp",
    verbosity = verbosity,
    tol_gap = tol_gap,
    tol_infeas = tol_infeas,
    line_search = line_search
)
}

fit_gapsafe <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "gap_safe",
    verbosity = verbosity,
    tol_gap = tol_gap,
    tol_infeas = tol_infeas,
    line_search = line_search
)

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
cat("gapsafe:\n")
cat("full = ", fit_gapsafe$full_time, "\n")
cat("passes = ", sum(fit_gapsafe$passes), "\n")

# plot(fit$passes, type = "l")
# lines(fit.w$passes, col = "red")
