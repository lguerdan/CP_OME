from attrdict import AttrDict

from data import loaders
from model import *
from numpy.random import permutation


###########################################################
######## Parameter estimation
###########################################################

def learn_parameters(ccpe_dataset, config, true_params):

    error_params_hat = AttrDict({})

    for do in config.target_POs:
        if config.learn_parameters == True:
            error_params_hat[f'alpha_{do}_hat'],  error_params_hat[f'beta_{do}_hat'] = crossfit_ccpe(ccpe_dataset, do, true_params, config)
        else:
            error_params_hat[f'alpha_{do}_hat'],  error_params_hat[f'beta_{do}_hat'] = true_params[f'alpha_{do}'],  true_params[f'beta_{do}']
    
    return error_params_hat


def crossfit_ccpe(ccpe_dataset, do, true_params, config):

    X_train, Y_train = ccpe_dataset['X_train'], ccpe_dataset['Y_train']

    shuffle_ix = permutation(Y_train.reset_index().index)
    X_train = X_train.iloc[shuffle_ix]
    Y_train = Y_train.iloc[shuffle_ix]

    split_ix = int(X_train.shape[0]*.7)

    ccpe_split_1 = AttrDict({
        'X_train': X_train[:split_ix],
        'Y_train': Y_train[:split_ix],
        'X_test': X_train[split_ix:],
        'Y_test': Y_train[split_ix:],
    })

    _, alpha_1_hat, beta_1_hat = ccpe_multiestimate(ccpe_split_1, do, true_params, config)
    
    return alpha_1_hat, beta_1_hat


def ccpe_multiestimate(dataset, do,  true_params, config):
    '''
        Fit class probability function and evaluate min/max on held-out data
    '''

    loss_config = AttrDict({
        'alpha': None,
        'beta':  None,
        'do': do,
        'reweight': False
    })

    train_loader, test_loader = loaders.get_loaders(
        X_train=dataset.X_train,
        YCF_train=dataset.Y_train,
        X_test=dataset.X_test,
        YCF_test=dataset.Y_test,
        target='Y', 
        do=do, 
        conditional=True
    )

    # Fit Y ~ X|T=t
    eta = MLP(n_feats=dataset.X_train.shape[1])
    losses = train(eta, train_loader, loss_config=loss_config, n_epochs=config.n_epochs, lr=config.lr,
        milestone=config.milestone, gamma=config.gamma, desc=f"CCPE: {do}")
    _, py_hat = evaluate(eta, test_loader)
    

    # Estimate error parameters via multiple methods     
    Y_test = dataset.Y_test
    D, pD, E, YS_0, YS_1, YS, Y = Y_test['D'],  Y_test['pD'], Y_test['E'], Y_test['YS_0'], Y_test['YS_1'], Y_test['YS'], Y_test['Y']

    # Baserate anchors
    E_YS = YS[(E==1) & (D==do)].mean()
    E_Y = Y[(E==1) & (D==do)].mean()

    if config.identification_pair == 'weak_seperability' or 'identification_pair' not in config:
        
        ci = py_hat.min()
        ci_s = 0
        cj = 1 - py_hat.max()
        cj_s = 1

        # Two anchor points available
        alpha_hat = (ci_s*cj- ci*cj_s)/(ci_s-cj_s)
        beta_hat = (ci*cj_s-ci+ci_s-cj_s+cj-ci_s*cj)/(ci_s-cj_s)


    elif config.identification_pair == 'baserate_min_anchor':

        ci = py_hat.min()
        ci_s = 0
        cj = E_Y
        cj_s = E_YS

        # Two anchor points available
        alpha_hat = (ci_s*cj- ci*cj_s)/(ci_s-cj_s)
        beta_hat = (ci*cj_s-ci+ci_s-cj_s+cj-ci_s*cj)/(ci_s-cj_s)

    elif config.identification_pair == 'baserate_max_anchor':

        ci = 1 - py_hat.max()
        ci_s = 1
        cj = E_Y
        cj_s = E_YS

        # Two anchor points available
        alpha_hat = (ci_s*cj- ci*cj_s)/(ci_s-cj_s)
        beta_hat = (ci*cj_s-ci+ci_s-cj_s+cj-ci_s*cj)/(ci_s-cj_s)

    elif config.identification_pair == 'beta_min_anchor':

        ci = py_hat.min()
        ci_s = 0
        beta = true_params[f'beta_{do}']
        
        # Closed form for alpha given beta
        alpha_hat = (ci - (1-beta)*ci_s)/(1-ci_s)
        beta_hat = beta

    else:
        raise Exception("Invalid param setting")
    
    print('===========================================')
    print(f'Error setting: {config.identification_pair}')
    print(f'Alpha hat: {alpha_hat}')
    print(f'Beta hat: {beta_hat}')

    print(f'Alpha: { true_params[f"alpha_{do}"]}')
    print(f'Beta: { true_params[f"beta_{do}"]}')
    print(f'Outcome baserate: {E_YS}')
    print('===========================================')

    return py_hat, alpha_hat, beta_hat


def ccpe(dataset, do, config):
    '''
        Fit class probability function and evaluate min/max on held-out data
    '''

    loss_config = AttrDict({
        'alpha': None,
        'beta':  None,
        'do': do,
        'reweight': False
    })

    train_loader, test_loader = loaders.get_loaders(
        X_train=dataset.X_train,
        YCF_train=dataset.Y_train,
        X_test=dataset.X_test,
        YCF_test=dataset.Y_test,
        target='Y', 
        do=do, 
        conditional=True
    )

    # Fit Y ~ X|T=t
    eta = MLP(n_feats=dataset.X_train.shape[1])
    losses = train(eta, train_loader, loss_config=loss_config, n_epochs=config.n_epochs, lr=config.lr,
        milestone=config.milestone, gamma=config.gamma, desc=f"CCPE: {do}")
    _, py_hat = evaluate(eta, test_loader)
    
    # Compute error parameters from predicted probabilities
    alpha_hat = py_hat.min()
    beta_hat = 1 - py_hat.max()

    return py_hat, alpha_hat, beta_hat
