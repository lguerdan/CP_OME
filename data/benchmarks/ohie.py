import pandas as pd
import numpy as np
from numpy.random import permutation


def generate_ohie_data(OHIE_PATH, error_params, train_ratio=.7, shuffle=True):
    
    ohie_df = pd.read_csv(OHIE_PATH)
    
    YS = ohie_df[['Y']].squeeze()
    D = ohie_df[['D']].squeeze()
    ohie_df.drop(columns=['Y', 'D'], inplace=True)

    YS_0 = np.zeros_like(YS)
    YS_1 = np.zeros_like(YS)

    YS_0[D==0] = YS[D==0]
    YS_1[D==1] = YS[D==1]

    Y_0 = YS_0.copy()
    Y_1 = YS_1.copy()
    Y = np.zeros_like(YS)

    alpha_0_errors = ((Y_0 == 0) & np.random.binomial(1, error_params['alpha_0'], size=ohie_df.shape[0]))
    alpha_1_errors = ((Y_1 == 0) & np.random.binomial(1, error_params['alpha_1'], size=ohie_df.shape[0]))
    beta_0_errors = ((Y_0 == 1) & np.random.binomial(1, error_params['beta_0'], size=ohie_df.shape[0]))
    beta_1_errors = ((Y_1 == 1) & np.random.binomial(1, error_params['beta_1'], size=ohie_df.shape[0]))

    Y_0[alpha_0_errors == 1] = 1
    Y_0[beta_0_errors == 1] = 0
    Y_1[alpha_1_errors == 1] = 1
    Y_1[beta_1_errors == 1] = 0

    Y[D==0] = Y_0[D==0]
    Y[D==1] = Y_1[D==1]

    dataset_y = {
        'YS': YS,
        'YS_0': YS_0,
        'YS_1': YS_1,
        'Y_0': Y_0,
        'Y_1': Y_1,
        'Y': Y,
        'pD': np.ones_like(D) * D.mean(),
        'pD_hat': np.ones_like(D) * D.mean(),
        'D': D,
        'E': np.ones_like(YS) # Include for computing the ATT on JOBS test data
    }

    X, Y = pd.DataFrame(ohie_df), pd.DataFrame(dataset_y)

    if shuffle: 
        suffle_ix = permutation(X.index)
        X = X.iloc[suffle_ix]
        Y = Y.iloc[suffle_ix]

    split_ix = int(X.shape[0]*train_ratio)
    X_train = X[:split_ix]
    X_test = X[split_ix:]
    Y_train = Y[:split_ix]
    Y_test = Y[split_ix:]

    #Selection bias: medicare opportunity not provided to individuals above the federal poverty line
    X_train_s = X_train[(Y_train['D'].to_numpy() == 0) |
        ((Y_train['D'].to_numpy() == 1) & (X_train['above_federal_pov'].to_numpy() == 0))]

    Y_train_s = Y_train[(Y_train['D'] == 0).to_numpy() | 
        ((Y_train['D'] == 1).to_numpy() & (X_train['above_federal_pov'] == 0).to_numpy())]  

    return X_train_s, X_test, Y_train_s, Y_test
