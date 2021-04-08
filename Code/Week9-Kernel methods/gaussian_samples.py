#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:36:24 2018

@author: paris
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from utilities import gen_Gaussian_samples, gen_plot_Gaussian_samples, plot_Gaussian_contours, plot_sample_dimensions, set_limits

if __name__ == "__main__":    
    
    #############################
    colors = ['r','g','b','m','k']
    markers = ['p','d','o','v','<']
    
    N=5 # Number of samples
    mu = np.array([0,0])  # Mean of the 2D Gaussian
    sigma = np.array([[1, 0.5], [0.5, 1]]); # covariance of the Gaussian
    
    # Generate samples
    samples = gen_Gaussian_samples(mu,sigma,N) 
    
    # Plot contours
    f = plt.figure(figsize=(12,12)); 
    ax1 = plt.subplot(1, 2, 1,autoscale_on=False, aspect='equal')
    set_limits(samples)
    plot_Gaussian_contours(samples[:,0:1],samples[:,1:2],mu,sigma)
    
    # Plot samples
    for i in np.arange(N):
        plt.plot(samples[i,0],samples[i,1], 'o', color=colors[i], marker=markers[i],ms=10)
    plt.grid()
    plt.gca().set_title(str(N) + ' samples of a bivariate Gaussian.')
    
    ax2=plt.subplot(1, 2, 2,autoscale_on=False, aspect='equal')
    plot_sample_dimensions(samples=samples, colors=colors, markers=markers)


    #############################
    # Plot with contours. Compare a correlated vs almost uncorrelated Gaussian
    sigmaUncor = np.array([[1, 0.02], [0.02, 1]]);
    sigmaCor = np.array([[1, 0.95], [0.95, 1]]);
    
    f = plt.figure(figsize=(15,15)); 
    
    ax=plt.subplot(1, 2, 1); ax.set_aspect('equal')
    samplesUncor = gen_Gaussian_samples(mu,sigmaUncor)
    plot_Gaussian_contours(samplesUncor[:,0],samplesUncor[:,1], mu, sigmaUncor)
    gen_plot_Gaussian_samples(mu, sigmaUncor)
    plt.gca().set_title('Weakly correlated Gaussian')
    
    ax=plt.subplot(1, 2, 2); ax.set_aspect('equal')
    samplesCor=gen_Gaussian_samples(mu,sigmaCor)
    plot_Gaussian_contours(samplesCor[:,0],samplesCor[:,1], mu, sigmaCor)
    gen_plot_Gaussian_samples(mu, sigmaCor)
    plt.gca().set_title('Stongly correlated Gaussian')
    
    
    #############################
    # But let's plot them as before dimension-wise...
    f = plt.figure(figsize=(18,5)); 
    perm = np.random.permutation(samplesUncor.shape[0])[0::14]
    
    ax1=plt.subplot(1, 2, 1); ax1.set_aspect('auto')
    plot_sample_dimensions(samplesUncor[perm,:])
    plt.gca().set_title('Weakly correlated')
    ax2=plt.subplot(1, 2, 2,sharey=ax1); ax2.set_aspect('auto')
    plot_sample_dimensions(samplesCor[perm,:])
    plt.gca().set_title('Strongly correlated')
    plt.ylim([samplesUncor.min()-0.3, samplesUncor.max()+0.3])
    
    
    #############################
    # Let's plot an 8-dimensional Gaussian...
    N=5
    mu = np.array([0,0,0,0,0,0,0,0])
    D = mu.shape[0]
    
    # Generate random covariance matrix
    tmp = np.sort(sp.random.rand(D))[:,None]
    tmp2 = tmp**np.arange(5)
    sigma = 5*np.dot(tmp2,tmp2.T) + 0.005*np.eye(D)
    
    samples = gen_Gaussian_samples(mu,sigma,N)
    
    for i in np.arange(N):
        plt.plot(tmp,samples[i,:], '-o')
    plt.grid()
    
    plt.gca().set_title(str(N) + ' samples of a ' + str(D) + ' dimensional Gaussian')
    
    
    #############################
    # Let's plot an 200-dimensional Gaussian...
    N=5
    D=200
    mu = np.zeros((D,1))[:,0]
    
    # Generate random covariance matrix
    tmp = np.sort(sp.random.rand(D))[:,None]
    tmp2 = tmp**np.arange(5)
    sigma = 5*np.dot(tmp2,tmp2.T)+ 0.0005*np.eye(D)
    
    samples = gen_Gaussian_samples(mu,sigma,N)
    
    for i in np.arange(N):
        plt.plot(tmp,samples[i,:], '-')
    
    plt.gca().set_title(str(N) + ' samples of a ' + str(D) + ' dimensional Gaussian')
    
    
    #############################
    # Let's plot an 2000-dimensional Gaussian...
    N=5
    D=2000
    mu = np.zeros((D,1))[:,0]
    
    # Generate random covariance matrix
    tmp = np.sort(sp.random.rand(D))[:,None]
    tmp2 = tmp**np.arange(5)
    sigma = 10*np.dot(tmp2,tmp2.T/0.02)+ 0.0001*np.eye(D)
    
    samples = gen_Gaussian_samples(mu,sigma,N)
    
    for i in np.arange(N):
        plt.plot(tmp,samples[i,:], '-')
    
    plt.gca().set_title(str(N) + ' samples of a ' + str(D) + ' dimensional Gaussian')