import time
import pandas as pd
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from celer import celer_path

readRDS = robjects.r['readRDS']

df = readRDS("data/simulated_data/1.rds")

X = pandas2ri.rpy2py_floatvector(df[0])
y = pandas2ri.rpy2py_floatvector(df[1])

n, p = X.shape

y -= np.mean(y)

alpha_max = np.max(np.abs(X.T.dot(y)))
n_alphas = 100

if n < p:
    alpha_min = 0.01
else:
    alpha_min = 1e-4

alphas = np.exp(
    np.linspace(np.log(alpha_max), np.log(alpha_max * alpha_min),
                n_alphas)) / n

t0 = time.time()
_, coefs, gaps, n_iter = celer_path(X,
                                    y,
                                    pb='lasso',
                                    alphas=alphas,
                                    tol=1295.11 * 1e-6 / n,
                                    return_n_iter=True,
                                    prune=True,
                                    verbose=0)
