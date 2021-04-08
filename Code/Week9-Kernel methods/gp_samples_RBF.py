#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:03:20 2018

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

#np.random.seed(1234)

if __name__ == "__main__":

    N = 1000
    D = 1
    lb = -10.0*np.ones(D)
    ub = 10.0*np.ones(D)
    jitter = 1e-8
    samples = 10

    # A simple vectorized rbf kernel
    def kernel(x,xp,hyp):
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:])
        diffs = np.expand_dims(x /lengthscales, 1) - \
                np.expand_dims(xp/lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))


    # Training data
    X = np.linspace(lb, ub, N)

    # Kernel hyper-parameters
    sigma_f = np.array([np.log(1.0)])
    lam = np.array([np.log(1.0)])
    hyp = np.concatenate([sigma_f, lam])

    # GP prior mean and covariance
    mu = 0.0
    K_xx = kernel(X, X, hyp)

    # Get cholesky decomposition (square root) of the
    # covariance matrix
    L = np.linalg.cholesky(K_xx + jitter*np.eye(N))
    # Sample 3 sets of standard normals for our test points,
    # multiply them by the square root of the covariance matrix
    f_prior = mu + np.dot(L, np.random.normal(size=(N,samples)))

    # Now let's plot the sampled functions.
    plt.figure(1)
    plt.plot(X, f_prior, linewidth = 2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('%d samples from a GP prior' % (samples))
    plt.show()
