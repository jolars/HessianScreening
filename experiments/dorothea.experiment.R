rm(list=ls())
graphics.off()
library(HessianScreening)
library(tidyr)
library(tibble)

d <- readRDS(file.path("data", paste0("dorothea", ".rds")))
X <- d$X
y <- d$y
family =  "gaussian"
screening_type = "hessian"
fit <- lassoPath(
    X,
    y,
    family = family,
    screening_type = screening_type,
    hessian_warm_starts=F,
    verbosity = 0
)
fit.w <- lassoPath(
    X,
    y,
    family = family,
    screening_type = "working",
    verbosity = 0
)
cat('***************\n')
cat('hessian:\n')
cat('full = ',fit$full_time,'\n')
cat('cd_time = ',sum(fit$cd_time),'\n')
cat('corr_time = ',sum(fit$corr_time),'\n')
cat('hess_time = ',sum(fit$hess_time),'\n')
cat('***************\n')
cat('working:\n')
cat('full = ',fit.w$full_time,'\n')
cat('cd_time = ',sum(fit.w$cd_time),'\n')
cat('corr_time = ',sum(fit.w$corr_time),'\n')


plot(fit$passes,type='l')
lines(fit.w$passes,col='red')
