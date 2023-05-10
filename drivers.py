import numpy as np
import pandas as pd
import numpy.matlib
from attrdict import AttrDict
import os, sys, torch, json

import ccpe, erm, model, utils
from data.benchmarks import synthetic, ohie, jobs


###########################################################
######## Risk minimmization experiments
###########################################################

def run_benchmark_risk_minimization_exp(config, baselines, param_configs, exp_name):
    '''
        Run each error parameter configuration over each benchmark environment. Keep sample size fixed. 
    '''

    exp_path = f'{config.log_dir}/{exp_name}/'
    utils.write_file(json.dumps(config), exp_path, f'config.json')

    for error_params in param_configs:
        te_results = []
        po_results = []
        
        for benchmark in config.benchmarks:
            config.benchmark = benchmark

            for run_num in range(config.n_runs):
            
                print('===============================================================================================================')
                print(f"Benchmark: {config.benchmark.name}, RUN: {run_num}, alpha_0: {error_params.alpha_0}, alpha_1: {error_params.alpha_1}, beta_0: {error_params.beta_0}, beta_1: {error_params.beta_1}")
                print('=============================================================================================================== \n')

                te_baseline_metrics, po_baseline_metrics = erm.run_model_comparison(config, baselines, error_params)
                te_results.extend(te_baseline_metrics)
                po_results.extend(po_baseline_metrics)

        po_df, te_df = pd.DataFrame(po_results), pd.DataFrame(te_results)
        utils.write_file(po_df, exp_path, f'runs={config.n_runs}_epochs={config.n_epochs}_alpha={error_params.alpha_0}_beta={error_params.beta_0}_PO.csv')
        utils.write_file(te_df, exp_path, f'runs={config.n_runs}_epochs={config.n_epochs}_alpha={error_params.alpha_0}_beta={error_params.beta_0}_TE.csv')

    return po_df, te_df

def run_param_assumption_risk_minimization_exp(config, baselines, param_configs, exp_name):
    '''
        Run each error parameter configuration over each benchmark environment. Keep sample size fixed. 
    '''

    exp_path = f'{config.log_dir}/{exp_name}/'
    utils.write_file(json.dumps(config), exp_path, f'config.json')

    for error_params in param_configs:
        te_results = []
        po_results = []
        
        for benchmark in config.benchmarks:
            config.benchmark = benchmark

            for identification_pair in config.assumptions:
                config.identification_pair = identification_pair
               
                for run_num in range(config.n_runs):
                
                    print('===============================================================================================================')
                    print(f"Benchmark: {config.benchmark.name}, RUN: {run_num}, alpha_0: {error_params.alpha_0}, alpha_1: {error_params.alpha_1}, beta_0: {error_params.beta_0}, beta_1: {error_params.beta_1}")
                    print('=============================================================================================================== \n')

                    te_baseline_metrics, po_baseline_metrics = erm.run_model_comparison(config, baselines, error_params)
                    te_results.extend(te_baseline_metrics)
                    po_results.extend(po_baseline_metrics)

        po_df, te_df = pd.DataFrame(po_results), pd.DataFrame(te_results)
        utils.write_file(po_df, exp_path, f'runs={config.n_runs}_epochs={config.n_epochs}_alpha={error_params.alpha_0}_beta={error_params.beta_0}_PO.csv')
        utils.write_file(te_df, exp_path, f'runs={config.n_runs}_epochs={config.n_epochs}_alpha={error_params.alpha_0}_beta={error_params.beta_0}_TE.csv')

    return po_df, te_df

def run_risk_minimization_exp(config, baselines, param_configs, exp_name):
    '''
        Vary sample size for synthetic benchmark environment. 
    '''

    exp_path = f'{config.log_dir}/{exp_name}/'
    utils.write_file(json.dumps(config), exp_path, f'config.json')
    
    for NS in config.sample_sizes:
        te_results = []
        po_results = []
        config.benchmark.NS = NS
        for error_params in param_configs:

            for run_num in range(config.n_runs):
            
                print('===============================================================================================================')
                print(f"NS: {NS}, RUN: {run_num}, alpha_0: {error_params.alpha_0}, alpha_1: {error_params.alpha_1}, beta_0: {error_params.beta_0}, beta_1: {error_params.beta_1}")
                print('=============================================================================================================== \n')

                te_baseline_metrics, po_baseline_metrics = erm.run_model_comparison(config, baselines, error_params, NS)
                te_results.extend(te_baseline_metrics)
                po_results.extend(po_baseline_metrics)

        po_df, te_df = pd.DataFrame(po_results), pd.DataFrame(te_results)

       
        utils.write_file(po_df, exp_path, f'runs={config.n_runs}_epochs={config.n_epochs}_benchmark={config.benchmark.name}_samples={NS}_PO.csv')
        utils.write_file(te_df, exp_path, f'runs={config.n_runs}_epochs={config.n_epochs}_benchmark={config.benchmark.name}_samples={NS}_TE.csv')

    return po_df, te_df

###########################################################
######## Parameter estimation experiments
###########################################################

def run_ccpe_exp(config, error_param_configs, sample_sizes, do=0):
    
    results = []
    for error_params in error_param_configs:
        for NS in sample_sizes:
            config.benchmark.update({'NS': NS})
            for RUN in range(config.n_runs):
                X_train, X_test, Y_train, Y_test = loader.get_benchmark(config.benchmark, error_params)
                split_ix = int(X.shape[0]*config.train_test_ratio)

                dataset = {
                    'X_train': X[:split_ix],
                    'Y_train': Y[:split_ix],
                    'X_test': X[split_ix:],
                    'Y_test': Y[split_ix:]
                }

                alpha_hat, beta_hat = ccpe_multiestimate(dataset, do, config)

                results.append({
                    'NS': NS,
                    'benchmark': config.benchmark.name,
                    'alpha': error_params[f'alpha_{do}'],
                    'beta': error_params[f'beta_{do}'],
                    'alpha_hat': alpha_hat,
                    'beta_hat': beta_hat,
                    'alpha_error': error_params[f'alpha_{do}'] - alpha_hat,
                    'beta_error': error_params[f'beta_{do}'] - beta_hat
                })

    ccpe_results = pd.DataFrame(results)
    path = f'{config.log_dir}/{exp_name}/'
    utils.write_file(ccpe_results, path, f'{config.log_dir}/parameter_estimation_runs={config.n_runs}_epochs={config.n_epochs}_benchmark={config.benchmark.name}.csv')

    return ccpe_results


if __name__ == '__main__':

    exp_type, exp_name = sys.argv[1], sys.argv[2]
    config = AttrDict(json.load(open(f'configs/{exp_name}.json')))

    if exp_type == 'erm':
        run_risk_minimization_exp(config, config.baselines, config.error_params, exp_name)

    if exp_type == 'erm_hyperparam':
        run_hyperparam_exp(config, config.baselines, config.error_params, exp_name)

    if exp_type == 'erm_semisyn':
        run_benchmark_risk_minimization_exp(config, config.baselines, config.error_params, exp_name)

    if exp_type == 'erm_semisyn_assumption':
        run_param_assumption_risk_minimization_exp(config, config.baselines, config.error_params, exp_name)
