import torch, utils
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score
from attrdict import AttrDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        if n_feats > 50:

            # OHIE model
            self.layers = nn.Sequential(
                nn.Linear(n_feats, 50),
                nn.ReLU(),
                nn.Linear(50, 30),
                nn.ReLU(),
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )

        elif n_feats > 5:

            # JOBS model
            self.layers = nn.Sequential(
                nn.Linear(n_feats, 30),
                nn.ReLU(),
                nn.Linear(30, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )

        else:

            # Synthetic model
            self.layers = nn.Sequential(
                nn.Linear(n_feats, 40),
                nn.Tanh(),
                nn.Linear(40, 30),
                nn.Tanh(),
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        y_pred = self.layers(x)
        return torch.clamp(y_pred, 1e-9, 1. - 1e-9)

def get_sample_weights(x, pd, loss_config, propensity_model):
    pd_hat = propensity_model(x) if propensity_model else pd
    return (1-loss_config.d_mean)/(1-pd_hat) if loss_config.do == 0 else loss_config.d_mean/pd_hat

def train(model, train_loader, loss_config, n_epochs, lr, milestone, gamma, desc, propensity_model=None):

    opt = optim.Adam(model.parameters(), lr=lr)
    epoch_loss = []
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[milestone], gamma=gamma)

    for epoch in tqdm(range(0, n_epochs), desc=desc):
        current_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            opt.zero_grad()
            x, y, pd, _ = data
            x, y, pd = x.to(DEVICE), y.to(DEVICE), pd.to(DEVICE)
            model = model.to(DEVICE)
            y_hat = model(x)
            balancing_weights = get_sample_weights(x, pd, loss_config, propensity_model) if loss_config.reweight else None
            loss = get_loss(y_hat, y, loss_config, weights=balancing_weights)
            loss.backward()
            opt.step()
            current_loss += loss.item()
        scheduler.step()

        epoch_loss.append(current_loss)
        current_loss = 0.0

    return epoch_loss

def evaluate(model, loader):

    preds = []
    labels = []
    avg_loss = 0
    loss = torch.nn.BCELoss()

        
    for i, data in enumerate(loader, 0):
        x, y, _, _ = data
        model = model.to(DEVICE)
        x, y = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        preds.append(outputs)
        labels.append(y)
        avg_loss += (1/len(loader)) * loss(outputs, y).item()

    # Compute potential outcome classification metrics
    py_hat = torch.cat(preds, dim=0).cpu().detach().numpy().squeeze()
    y = torch.cat(labels, dim=0).cpu().detach().numpy()
    y_hat = np.zeros_like(y)
    y_hat[py_hat > .5] = 1

    metrics = AttrDict({
        'AU-ROC': roc_auc_score(y, py_hat),
        'ACC': (y_hat == y).mean(),
        'loss': avg_loss
    })

    return metrics, py_hat


def get_loss(py_hat, y, loss_config, weights):
    '''Surrogate loss parameterized by alpha, beta'''

    loss = torch.nn.BCELoss(reduction='none')
    alpha, beta = loss_config['alpha'], loss_config['beta']
    if alpha != None and beta != None:
        phat_y1 = py_hat[y==1]
        phat_y0 = py_hat[y==0]

        try:
            denom = max(1-beta-alpha, .01)
            y1_losses = ((1-alpha)*loss(phat_y1, torch.ones_like(phat_y1)) -
            beta*loss(phat_y1, torch.zeros_like(phat_y1))) / denom

            y0_losses = ((1-beta)*loss(phat_y0, torch.zeros_like(phat_y0)) -
            alpha*loss(phat_y0, torch.ones_like(phat_y0))) / denom
            val_loss = torch.cat([y1_losses, y0_losses])
        
        except:
            print('estiamted alpha', alpha)
            print('estimated beta', beta)
            print('py_hat', py_hat)
            print(f'phat_y1 min: {phat_y1.min()}, phat_y1 max: {phat_y1.max()}')
            print(f'phat_y0 min: {phat_y0.min()}, phat_y0 max: {phat_y0.max()}')
            val_loss = np.array([0]) # Temporary fix so training doesn't break

    else:
        val_loss = loss(py_hat, y)

    if weights != None:
        val_loss = val_loss*weights

    return val_loss.mean()
