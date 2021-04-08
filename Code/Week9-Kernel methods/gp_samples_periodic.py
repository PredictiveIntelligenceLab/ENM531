#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:03:20 2018

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

if __name__ == "__main__":    
    
    N = 256
    D = 1
    lb = -10.0*np.ones(D)
    ub = 10.0*np.ones(D)
    jitter = 1e-14
    samples = 5
    domain_L = 10.0
    
    # A simple vectorized periodic kernel
    def kernel(x, xp, hyp):     
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:-1])  
        period = np.exp(hyp[-1])   
        N = x.shape[0]
        N_prime = xp.shape[0]    
        x = np.matmul(x, np.ones((1,N_prime)))
        xp = np.matmul(np.ones((N,1)),xp.T)        
        K = output_scale*np.exp(-2.0*np.sin(np.pi*np.abs(x-xp)/period)**2/lengthscales**2)
        return K
    
    # Training data    
    X = np.linspace(lb, ub, N)
    
    # Kernel hyper-parameters    
    sigma_f = np.array([np.log(0.1)])
    lam = np.array([np.log(1.0)])
    p = np.array([np.log(domain_L)])
    hyp = np.concatenate([sigma_f, lam, p])
    
    # GP prior mean and covariance
    mu = 0.0
    K_xx = kernel(X, X, hyp)

    # Compute the Cholesky factors
    L = np.linalg.cholesky(K_xx + jitter*np.eye(N))
    # Generate samples
    f_prior = mu + np.dot(L, np.random.normal(size=(N,samples)))
    
    # Plot the sampled functions.
    plt.figure(1)
    plt.plot(X, f_prior, linewidth = 2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('%d samples from a GP prior' % (samples))
    plt.show()