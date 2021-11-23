import blitzl1
import numpy as np
import time

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

verbose = False

readRDS = robjects.r['readRDS']

df = readRDS("data/cadata.rds")

X = pandas2ri.rpy2py_floatvector(df[0])
y = pandas2ri.rpy2py_floatvector(df[1])

n_lambda = 100

n, p = np.shape(X)

# standardize
y = y - np.mean(y)

means = np.zeros(p)
stddev = np.zeros(p)

for j in range(p):
    means[j] = np.mean(X[:, j])
    stddev[j] = np.std(X[:, j])
    X[:, j] = (X[:, j] - means[j]) / stddev[j]

prob = blitzl1.LassoProblem(X, y)

lambda_max = prob.compute_lambda_max()

lambda_min_ratio = 1e-2 if p > n else 1e-4

lambdas = np.exp(
    np.linspace(np.log(lambda_max), np.log(lambda_min_ratio * lambda_max),
                n_lambda))

dev_ratios = np.zeros(n_lambda)

residual = -y
null_dev = np.linalg.norm(y)**2

dev_prev = null_dev
beta_hat = None

blitzl1.set_verbose(verbose)
blitzl1.set_tolerance(1e-3)

start_time = time.monotonic()

for i in range(n_lambda):
    # print("i: ", i, ", lambda: ", lambdas[i])
    sol = prob.solve(lambdas[i], initial_x=beta_hat)

    beta_hat = sol.x
    residual = X @ beta_hat - y
    dev = np.linalg.norm(X @ beta_hat - y)**2
    dev_ratio = 1 - dev / null_dev
    dev_ratios[i] = dev_ratio
    dev_change = 1 - dev / dev_prev
    n_active = np.sum(beta_hat != 0)
    dev_prev = dev

    if (i > 0 and dev_change <= 1e-5) or dev_ratio >= 0.999 or (n <= p and
                                                                n_active > n):
        break

full_time = time.monotonic() - start_time

dev_ratios
