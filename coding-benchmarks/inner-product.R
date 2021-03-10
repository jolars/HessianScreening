library(bench)

n <- 100000
p <- 400
m <- 10

mark(
    HessianScreening:::denseInnerProduct(0, n, p, m),
    HessianScreening:::denseInnerProduct(1, n, p, m),
    HessianScreening:::denseInnerProduct(2, n, p, m)
)

n <- 1000
p <- 400000
m <- 1000
density <- 0.01

mark(
    HessianScreening:::sparseInnerProduct(0, n, p, m, density),
    HessianScreening:::sparseInnerProduct(1, n, p, m, density),
    HessianScreening:::sparseInnerProduct(2, n, p, m, density, 1),
    HessianScreening:::sparseInnerProduct(2, n, p, m, density, 8)
)
