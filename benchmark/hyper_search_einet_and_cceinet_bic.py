
from sklearn import datasets
import torch

from simple_einet.distributions import CCRatNormal, RatNormal
from simple_einet.einet import Einet, CCLEinet, EinetConfig

import numpy as np

import optuna

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


SEED = 1
EPOCHS = 50
TRIALS = 50
CLASS_IDX = -1

K_MIN = 1
K_MAX = 7
R_MIN = 1
R_MAX = 7
LR_MIN = 0.001
LR_MAX = 1

DIGITS = datasets.load_digits()
DATA = np.hstack((DIGITS.data, np.expand_dims(DIGITS.target, axis=1)))
NUM_FEATURES = 64
NUM_CLASSES = 10

device = torch.device("cpu")
torch.manual_seed(SEED)


def BIC(mean_nll, num_params, num_samples):
    return 2 * num_samples * mean_nll + np.log(num_samples) * num_params


def accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(-1)
        correct = (y == predictions).sum()
        total = y.shape[0]
        acc = correct / total * 100
        return acc


def score_cceinet(train_data, D=1, K=3, R=3, num_classes=NUM_CLASSES, class_idx=CLASS_IDX, dropout=0.0, epochs=EPOCHS, lr=0.2, in_eval=False):
    """ returning lists with test acc, train acc and train loss """

    train_data = torch.tensor(train_data).long().to(device)

    X_train = train_data[:, :class_idx]
    y_train = train_data[:, class_idx]

    config = EinetConfig(
        num_features=NUM_FEATURES,
        num_channels=D,
        num_sums=K,
        num_leaves=K,
        num_repetitions=R,
        cross_product=True,
        num_classes=num_classes,
        leaf_type=CCRatNormal,
        dropout=dropout)
    model = CCLEinet(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss = None

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train, y_train)
        loss = -outputs.mean()

        loss.backward()
        optimizer.step()

    if in_eval:
        return accuracy(model, X_train, y_train), loss, num_params
    return BIC(loss.item(), num_params, train_data.shape[0])


def score_einet(train_data, D=1, K=3, R=3, num_classes=NUM_CLASSES, class_idx=CLASS_IDX, dropout=0.0, epochs=EPOCHS, lr=0.07, in_eval=False):

    train_data = torch.tensor(train_data).long().to(device)

    X_train = train_data[:, :class_idx]
    y_train = train_data[:, class_idx]

    config = EinetConfig(
        num_features=NUM_FEATURES,
        num_channels=D,
        num_sums=K,
        num_leaves=K,
        num_repetitions=R,
        cross_product=True,
        num_classes=num_classes,
        leaf_type=RatNormal,
        leaf_kwargs={},
        dropout=dropout)
    model = Einet(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    nll = torch.nn.NLLLoss(weight=torch.ones(10) / 10)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss = None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        lls = model(X_train)
        loss = nll(lls, y_train)

        loss.backward()
        optimizer.step()

    if in_eval:
        return accuracy(model, X_train, y_train), loss, num_params

    return BIC(loss.item(), num_params, train_data.shape[0])


def score_einet_proxy(trial, in_eval=False):
    K = trial.suggest_int("K", K_MIN, K_MAX)
    R = trial.suggest_int("R", R_MIN, R_MAX)
    lr = trial.suggest_float("lr", LR_MIN, LR_MAX)

    return score_einet(DATA, K=K, R=R, lr=lr, in_eval=in_eval)


def score_cceinet_proxy(trial, in_eval=False):
    K = trial.suggest_int("K", K_MIN, K_MAX)
    R = trial.suggest_int("R", R_MIN, R_MAX)
    lr = trial.suggest_float("lr", LR_MIN, LR_MAX)

    return score_cceinet(DATA, K=K, R=R, lr=lr, in_eval=in_eval)


if __name__ == "__main__":
    einet_study = optuna.create_study(direction="minimize")
    einet_study.optimize(score_einet_proxy, n_trials=TRIALS)

    cceinet_study = optuna.create_study(direction="minimize")
    cceinet_study.optimize(score_cceinet_proxy, n_trials=TRIALS)

    print("einet best params:", einet_study.best_params)
    print("einet best value:", einet_study.best_value)
    print("einet eval:", score_einet_proxy(
        einet_study.best_trial, in_eval=True))

    print("cceinet best params:", cceinet_study.best_params)
    print("cceinet best value:", cceinet_study.best_value)
    print("cceinet eval:", score_cceinet_proxy(
        cceinet_study.best_trial, in_eval=True))

    # digits
    # einet best params: {'K': 7, 'R': 6, 'lr': 1.4350907130085637}
    # einet best value: 1309129.3460619296
    # einet eval: (tensor(79.2988), tensor(393.7412, grad_fn=<NllLossBackward0>), 8376)
    # cceinet best params: {'K': 7, 'R': 1, 'lr': 0.99254923282291}
    # cceinet best value: 1067786.1432766588
    # cceinet eval: (tensor(97.1063), tensor(293.1933, grad_fn=<NegBackward0>), 9020)
