library(bench)
library(Matrix)

n <- 10000
p <- 50000

X <- rsparsematrix(n,p, 0.01)

X_mean_scaled <- colMeans(X)

ind <- sort(sample(p, round(p*0.2))) - 1

mark(
    HessianScreening:::innerProductStandardized(0, X, ind, X_mean_scaled),
    HessianScreening:::innerProductStandardized(1, X, ind, X_mean_scaled)
)
