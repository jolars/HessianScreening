library(HessianScreening)

family <- "gaussian"

d <- readRDS(file.path("data", paste0("e2006-tfidf", ".rds")))
#d <- generateDesign(200, 20000, family = family, rho = 0.5, snr = 1)
X <- d$X
y <- d$y
n <- nrow(X)
p <- ncol(X)
verbosity <- 0
line_search <- 2
tol_gap <- 1e-5
tol_infeas <- 1e-4

# sparsity <- 1 - Matrix::nnzero(X) / length(X)
# sparsity * min(n, p) / max(n, p)

# n / max(n, p) * sparsity
fit_celer <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "celer",
    verbosity = verbosity,
    tol_gap = tol_gap,
    tol_infeas = tol_infeas,
    line_search = line_search,
    log_hessian_update_type = "full"
)

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

fit_celer$full_time
fit_hessian$full_time
fit_working$full_time

# X <- scale(X, scale=apply(X,2, function(x) sqrt((length(x)-1)/length(x))*sd(x)))
# y = y-mean(y)
# sim_dat = list(X=X, y=y)
# save(sim_dat,  file = "data/simpleXy.rda")
