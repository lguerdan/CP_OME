from attrdict import AttrDict
from sklearn.metrics import roc_auc_score

import data.loaders as loader
import ccpe, utils
from model import *

def run_model_comparison(config, baselines, error_params, NS=None):

    X_train, X_test, Y_train, Y_test = loader.get_benchmark(config.benchmark, error_params, NS)

    crossfit_erm_preds = { baseline.model: {} for baseline in baselines }

    if config.crossfit_erm == True:
        split_permuations = [(0,1,2), (0,2,1), (2,0,1)]
    else:
        split_permuations = [(0,1,2)]
    
    for s_ix, (p,q,r) in enumerate(split_permuations):

        data_splits = loader.get_splits(X_train, X_test, Y_train, Y_test, config)
        weight_dataset, ccpe_dataset, erm_dataset = data_splits[p], data_splits[q], data_splits[r]

        if config.learn_weights:
            propensity_model = learn_weights(weight_dataset, config)

        error_params_hat = ccpe.learn_parameters(ccpe_dataset, config, true_params=error_params)

        for baseline in baselines:

            # Use test data baesrate because D is experimentally assigned in test data
            loss_config = AttrDict({
                'd_mean': Y_test[Y_test['E'] == 1]['D'].mean(),
                'reweight': baseline.reweight,
            })
            baseline.propensity_model = propensity_model if config.learn_weights else None
            baseline.error_params_hat = error_params_hat

            preds, metrics =  run_erm_split(
                erm_dataset=erm_dataset,
                baseline_config=baseline,
                loss_config=loss_config,
                exp_config=config
            )
            crossfit_erm_preds[baseline.model][s_ix] = preds

    log_metadata = AttrDict({**error_params, **error_params_hat})
    log_metadata.benchmark = config.benchmark.name
    log_metadata.assumption = config.identification_pair if 'identification_pair' in config else 'weak_seperability'
    log_metadata.NS = NS
    log_metadata.val_loss = metrics.loss
    return compute_crossfit_metrics(crossfit_erm_preds, Y_test, len(split_permuations), config, log_metadata)


def run_erm_split(erm_dataset, baseline_config, loss_config, exp_config):
    '''
        erm_split: [erm train split (.33 of training data), ERM test split (test data)]    
    '''

    po_preds = {}

    for do in exp_config.target_POs:
        loss_config.alpha = baseline_config.error_params_hat[f'alpha_{do}_hat'] if baseline_config.sl else None
        loss_config.beta = baseline_config.error_params_hat[f'beta_{do}_hat'] if baseline_config.sl else None
        loss_config.do = do

        train_loader, test_loader = loader.get_loaders(
            X_train=erm_dataset.X_train,
            YCF_train=erm_dataset.Y_train,
            X_test=erm_dataset.X_test,
            YCF_test=erm_dataset.Y_test,
            target=baseline_config.target, 
            do=do,
            conditional=baseline_config.conditional
        )

        eta_model = MLP(n_feats=erm_dataset.X_train.shape[1])
        propensity_model = baseline_config.propensity_model if exp_config.learn_weights else None
        losses = train(eta_model, train_loader, loss_config=loss_config, n_epochs=exp_config.n_epochs,
            lr=exp_config.lr, milestone=exp_config.milestone, gamma=exp_config.gamma, desc=f"ERM: {baseline_config.model}")
        val_metrics, py_hat = evaluate(eta_model, test_loader)        
        po_preds[do] = py_hat
    
    return po_preds, val_metrics

def learn_weights(weight_dataset, config):
    '''
        Estimate weighting function on training dataset and run inference on evaluation fold
    '''

    train_loader, test_loader = loader.get_loaders(
        X_train=weight_dataset.X_train,
        YCF_train=weight_dataset.Y_train,
        X_test=weight_dataset.X_test,
        YCF_test=weight_dataset.Y_test,
        target='D', 
        do=0, 
        conditional=False
    )

    loss_config = AttrDict({
        'pd': weight_dataset.Y_train['D'].mean(),
        'reweight': False,
        'alpha': None,
        'beta': None
    })
        
    pi = MLP(n_feats=weight_dataset.X_train.shape[1])
    losses = train(pi, train_loader, loss_config=loss_config, n_epochs=15, lr=config.lr, milestone=config.milestone, gamma=config.gamma, desc='Propensity model')
    print('Weight losses: ', losses)
    return pi

def compute_crossfit_metrics(crossfit_erm_preds, Y_test, n_splits, config, log_metadata):
    
    te_metrics = []
    po_metrics = []
    
    for baseline_name, results in crossfit_erm_preds.items():

        po_preds = {}
        for do in config.target_POs:

            y = Y_test[f'YS_{do}']
            po_preds[do] = np.zeros_like(y)
            
            # Compute aggregate model prediction on evaluation fold
            for split in range(n_splits): 
                po_preds[do] = np.add(po_preds[do], (1/n_splits)*results[split][do])
                y_hat = po_preds[do] > .5 # threshold on Bayes' optimal classifier

                po_result = {
                    'AU-ROC': roc_auc_score(y, po_preds[do]),
                    'ACC': (y_hat == y).mean(),
                    'do': do,
                    'baseline': baseline_name
                }

            po_metrics.append({**log_metadata, **po_result})

        if len(config.target_POs) == 2:
            print(f'Baseline: {baseline_name}')
            te_result = compute_treatment_metrics(po_preds, Y_test, config.benchmark.name)
            te_metrics.append({**log_metadata, **te_result, 'baseline': baseline_name })
            
    return te_metrics, po_metrics


def compute_treatment_metrics(po_preds, Y_test, benchmark):

    D, pD, E, YS_0, YS_1, YS = Y_test['D'],  Y_test['pD'], Y_test['E'], Y_test['YS_0'], Y_test['YS_1'], Y_test['YS']
    YS_0_hat, YS_1_hat = po_preds[0], po_preds[1]

    if 'synthetic' in benchmark:
        ate = YS_1.mean() - YS_0.mean()

    elif benchmark == 'ohie':
        ate = 0.0158

    elif benchmark == 'jobs':
        # E[unemployment|program] - E[unemployment|no program]
        ate = -0.0779

    else: 
        raise Exception("Invalid benchmark")

    # Evaluate over factual and counterfactual outcomes
    # E=1 is required for experimental sub-sample of NSW study
    ate_hat = YS_1_hat[E==1].mean() - YS_0_hat[E==1].mean()
    print('ATE estimate: ', ate_hat)
    print(f'Y0: {YS_0[(E==1) & (D==0)].mean():.4}, Y0 hat: {YS_0_hat[E==1].mean():.4}')
    print(f'Y1: {YS_1[(E==1) & (D==1)].mean():.4}, Y1 hat: {YS_1_hat[E==1].mean():.4}')
    
    ate_metrics = {
        'ate': ate,
        'ate_hat': ate_hat,
        'ate_error': ate-ate_hat,
    }
    
    return {**ate_metrics}
