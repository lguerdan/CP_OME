import numpy as np
from numpy.random import permutation
import pandas as pd
import math
from attrdict import AttrDict

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
sns.set_style("ticks")
sns.set_context("poster")

def pi(x, func):
    if func=='uniform': 
        return .5*np.ones(x.shape[0])
        
    elif func=='linear': 
        return (.35 * x + .5).squeeze()

    elif func=='6cov_linear':
        return .6*x.mean(axis=1) +.1
    
    elif func=='6cov_linsep':
        return .3*(1-np.abs(x[:,0]-x[:,1])) + .2

def eta(x, environment):

    if environment=='sinusoid':
        return (.5 + .5 * np.sin(2.9*x + .1)).squeeze()


    elif environment=='piecewise_sinusoid':
        return np.piecewise(x,[
            ((-1 <= x) & (x <= -.237)),
            ((-.237 < x) & (x <= 1))], 
            [lambda v: .4+.4*np.cos(9*v+4.4), 
            lambda v: .546+.3*np.sin(8*v+.9)+.15*np.sin(10*v+.2)+.05*np.sin(30*v+.2)]).squeeze()

    elif environment=='low_base_rate_sinusoid':
        return (.5-.5 * np.sin(2.9*x+.1)).squeeze()

    else: 
        return np.piecewise(x,[
            ((-1 <= x) & (x <= -.5)),
            ((-.5 < x) & (x <= 0.2069)),
            ((0.2069 < x) & (x <= 0.8)),
            ((0.8 < x) & (x <= 1))],  
            [lambda v: -1.5*v-.75, 
             lambda v: 1.4*v+.7,
             lambda v: -1.5*v+1.3,
             lambda v: 1.25*v - .9 ])

def generate_syn_data(env, error_params, NS, train_ratio=.7, shuffle=True):  
    
    NS = NS if NS else env['NS']

    if env['name'] == 'synthetic_1D_sinusoidal':
        x = np.random.uniform(low=-1, high=1, size=(NS, 1))
    
    elif env['name'] == 'synthetic_2D_linsep':
        x = np.random.uniform(low=0, high=1, size=(NS, 2))

    elif env['name'] == 'synthetic_6D_shalt':
        x = np.random.uniform(low=0, high=1, size=(NS, 6))
        
    eta_star_0 = eta(x, environment=env['config']['Y0_PDF'])
    eta_star_1 = eta(x, environment=env['config']['Y1_PDF'])

    # Sample from target potential outcome class probability distributions
    YS_0 = np.random.binomial(1, eta_star_0, size=NS)
    YS_1 = np.random.binomial(1, eta_star_1, size=NS)
    
    Y_0 = YS_0.copy()
    Y_1 = YS_1.copy()

    alpha_0_errors = ((Y_0 == 0) & np.random.binomial(1, error_params['alpha_0'], size=NS))
    alpha_1_errors = ((Y_1 == 0) & np.random.binomial(1, error_params['alpha_1'], size=NS))
    beta_0_errors = ((Y_0 == 1) & np.random.binomial(1, error_params['beta_0'], size=NS))
    beta_1_errors = ((Y_1 == 1) & np.random.binomial(1, error_params['beta_1'], size=NS))

    Y_0[alpha_0_errors == 1] = 1
    Y_0[beta_0_errors == 1] = 0
    Y_1[alpha_1_errors == 1] = 1
    Y_1[beta_1_errors == 1] = 0

    # Apply consistency assumption to observe potential outcomes
    YS = np.zeros(NS, dtype=np.int64)
    Y = np.zeros_like(Y_0)

    pD = pi(x, func=env['config']['PI_PDF'])
    D = np.random.binomial(1, pD, size=NS)
    YS[D==0] = YS_0[D==0]
    YS[D==1] = YS_1[D==1]

    Y[D==0] = Y_0[D==0]
    Y[D==1] = Y_1[D==1]
        
    dataset_y = {
        'pYS_0': eta_star_0,
        'pYS_1': eta_star_1,
        'YS_0': YS_0,
        'YS_1': YS_1,
        'Y_0': Y_0,
        'Y_1': Y_1,
        'Y': Y,
        'pD': pD,
        'pD_hat': pD,
        'D': D,
        'YS': YS,
        'E': np.ones_like(YS) # Include for computign the ATT on JOBS test data
    }

    X, Y = pd.DataFrame(x), pd.DataFrame(dataset_y)
   
    if shuffle: 
        suffle_ix = permutation(X.index)
        X = X.iloc[suffle_ix]
        Y = Y.iloc[suffle_ix]

    split_ix = int(X.shape[0]*train_ratio)
    X_train = X[:split_ix]
    X_test = X[split_ix:]
    Y_train = Y[:split_ix]
    Y_test = Y[split_ix:]

    return X_train, X_test, Y_train, Y_test

def plot_syn_setup(path):

    benchmark =  AttrDict({
        'name': 'synthetic_1D_sinusoidal',
        'NS': 25000,
        'config': {
            'Y0_PDF': 'piecewise_sinusoid',
            'Y1_PDF': 'low_base_rate_sinusoid',
            'PI_PDF': 'linear'
            }
    })

    error_params = AttrDict({
        'alpha_0': 0.3,
        'alpha_1': 0,
        'beta_0': 0.1,
        'beta_1': 0 
    })

    X_train, X_test, Y_train, Y_test = generate_syn_data(benchmark, error_params, 30000)

    x = X_test.T.to_numpy().squeeze()
    inds = np.argsort(x)
    ys0sorted = Y_test['pYS_0'].to_numpy()[inds]
    ys1sorted = Y_test['pYS_1'].to_numpy()[inds]
    pd = Y_test['pD'].to_numpy()[inds]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 6.5), sharex=True)
    axes[0].plot(x[inds], ys0sorted, label='$\eta^*_0(x)$', color='darkblue')
    axes[0].plot(x[inds], 1-pd, label='$1-\pi(x)$', color='black', linestyle='dashed')
    axes[0].set_ylabel('$P$')
    axes[0].legend(bbox_to_anchor=(1.1, 1.45), ncol=2)


    axes[1].plot(x[inds], ys1sorted, label='$\eta^*_1(x)$', color='orange')
    axes[1].plot(x[inds], pd, label='$\pi(x)$', color='black', linestyle='dotted')
    axes[1].set_xlabel('$X$')
    axes[1].set_ylabel('$P$')
    axes[1].legend(bbox_to_anchor=(1.1, -0.35), ncol=2)

    plt.savefig(path, bbox_inches = 'tight', dpi=500)
