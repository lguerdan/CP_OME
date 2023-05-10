from attrdict import AttrDict
import torch

from data.benchmarks import synthetic, ohie, jobs

def get_benchmark(benchmark_config, error_params, NS):

    if 'synthetic' in benchmark_config['name']:
        return synthetic.generate_syn_data(benchmark_config, error_params, NS)

    elif benchmark_config['name'] == 'ohie':
        return ohie.generate_ohie_data(benchmark_config['path'], error_params)

    elif benchmark_config['name'] == 'jobs':
        return jobs.generate_jobs_data(benchmark_config, error_params)


def get_splits(X_train, X_test, Y_train, Y_test, config):
    
    N_train = X_train.shape[0]
    
    if not config.split_erm:
        return [AttrDict({
            'X_train': X_train.copy(deep=True),
            'X_test': X_test.copy(deep=True),
            'Y_train': Y_train.copy(deep=True),
            'Y_test': Y_test.copy(deep=True)
        }) for i in range(3)]
        
    else:
        split_ix_1, split_ix_2 = int(.33*N_train), int(.66*N_train)

        split1 = AttrDict({
            'X_train': X_train.iloc[:split_ix_1, :],
            'X_test': X_test,
            'Y_train': Y_train.iloc[:split_ix_1, :],
            'Y_test': Y_test
        })

        split2 = AttrDict({
            'X_train': X_train.iloc[split_ix_1:split_ix_2, :],
            'X_test': X_test,
            'Y_train': Y_train.iloc[split_ix_1:split_ix_2, :],
            'Y_test': Y_test
        })

        split3 = AttrDict({
            'X_train': X_train.iloc[split_ix_2:, :],
            'X_test': X_test,
            'Y_train': Y_train.iloc[split_ix_2:, :],
            'Y_test': Y_test
        })
        
        return [split1, split2, split3]

def get_loaders(X_train, YCF_train, X_test, YCF_test, target, do, conditional):

    X_train = X_train.copy(deep=True)
    YCF_train = YCF_train.copy(deep=True)
    X_test = X_test.copy(deep=True)
    YCF_test = YCF_test.copy(deep=True)

    if conditional:
        X_train = X_train[YCF_train['D']==do]
        YCF_train = YCF_train[YCF_train['D']==do]

    eval_target = 'D' if target == 'D' else f'YS_{do}'

    X_train = X_train.to_numpy()
    Y_train = YCF_train[target].to_numpy()[:, None]
    pD_train = YCF_train['pD'].to_numpy()[:, None]
    D_train = YCF_train['D'].to_numpy()[:, None]

    X_test = X_test.to_numpy()
    Y_test = YCF_test[eval_target].to_numpy()[:, None]
    pD_test = YCF_test['pD'].to_numpy()[:, None]
    D_test = YCF_test['D'].to_numpy()[:, None]

    print('****** Data info ********')
    print(f'DO: {do}')
    print('N Test: ', Y_test.shape[0])
    print('N Train: ', Y_train.shape[0])
    print('N[Y] train: ', Y_train.sum())
    print('N[Y] test: ', Y_test.sum())
    print('**************************')
    
    train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train), torch.Tensor(pD_train), torch.Tensor(D_train))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
    test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test), torch.Tensor(pD_test), torch.Tensor(D_test))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=1)
    
    return train_loader, test_loader
