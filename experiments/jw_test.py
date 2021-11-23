#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:52:33 2021

@author: jonaswallin
"""

import time
import pandas as pd
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path
from sklearn.datasets import fetch_openml

from celer import celer_path

def generate_X(n,p,s, rho=0.5):
    
    beta = np.zeros(p)
    beta[:s] = 1
    inds = np.linspace(1,p,p)
    Sigma = rho**(np.abs(np.subtract.outer(inds,inds)))
    L = npl.cholesky(Sigma)
    X= np.random.normal(size=(n,p))
    X = np.dot(X, L.T)
    sigma = np.sqrt( np.dot(beta.T, np.dot(Sigma, beta))) 
    y = np.dot(X,beta) + sigma * np.random.normal(size=n)
    return X, y, beta
    
import rdata
from sklearn import preprocessing
parsed = rdata.parser.parse_file("../../HessianScreening/data/simpleXy.rda")
converted = rdata.conversion.convert(parsed)

#X, y, beta_t = generate_X(n=1000,p=100,s=3,rho=0.8)
X = np.asfortranarray(converted['sim_dat']['X'])
n,p = X.shape
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)
y = converted['sim_dat']['y']
#y -=  np.mean(y)
n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) 
n_alphas = 100
if n < p:
    alpha_min = 0.01
else:
    alpha_min=1e-4

alphas = np.exp(np.linspace(np.log(alpha_max),np.log(alpha_max*alpha_min), n_alphas))/n_samples
#alphas = alphas[:43]
t0 = time.time()
_, coefs, gaps, n_iter, n_count = celer_path(
X, y, pb='lasso', alphas=alphas, tol= 1295.11*1e-6/n_samples,return_n_iter=True, prune=True, verbose= 0)
print('Celer time: {}'.format(time.time() - t0))
print(n_count)
#print(np.sum((y-np.dot(X,coefs[:,87]))**2) + alphas[87]*n_samples * sum(np.abs(coefs[:,87])))