
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
K_FOLDS = 5
EPOCHS = 100
TRIALS = 100

IRIS = datasets.load_iris()
DATA = np.hstack((IRIS.data, np.expand_dims(IRIS.target, axis=1)))

device = torch.device("cpu")
torch.manual_seed(SEED)


def accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(-1)
        correct = (y == predictions).sum()
        total = y.shape[0]
        acc = correct / total * 100
        return acc


def score_one_cceinet(train_data, test_data, D=1, K=3, R=3, num_classes=3, class_idx=4, dropout=0.0, epochs=40, lr=0.2):
    """ returning lists with test acc, train acc and train loss """

    X_train = train_data[:, :class_idx]
    X_test = test_data[:, :class_idx]
    y_train = train_data[:, class_idx].long()
    y_test = test_data[:, class_idx].long()

    config = EinetConfig(
        num_features=4,
        num_channels=D,
        num_sums=K,
        num_leaves=K,
        num_repetitions=R,
        num_classes=num_classes,
        leaf_type=CCRatNormal,
        dropout=dropout)
    model = CCLEinet(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    test_acc_list = list()

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train, y_train)
        loss = -outputs.mean()

        loss.backward()
        optimizer.step()

        model.eval()
        acc_test = accuracy(model, X_test, y_test)
        model.train()

        test_acc_list.append(acc_test.item())

    return test_acc_list


def score_one_einet(train_data, test_data, D=1, K=3, R=3, num_classes=3, class_idx=4, dropout=0.0, epochs=40, lr=0.07):

    X_train = train_data[:, :class_idx]
    X_test = test_data[:, :class_idx]
    y_train = train_data[:, class_idx].long()
    y_test = test_data[:, class_idx].long()

    config = EinetConfig(
        num_features=4,
        num_channels=D,
        num_sums=K,
        num_leaves=K,
        num_repetitions=R,
        num_classes=num_classes,
        leaf_type=RatNormal,
        leaf_kwargs={},
        dropout=dropout)
    model = Einet(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    test_acc_list = list()

    cross_entropy = torch.nn.NLLLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = cross_entropy(outputs, y_train)
        loss.backward()
        optimizer.step()

        acc_test = accuracy(model, X_test, y_test)

        test_acc_list.append(acc_test.item())

    return test_acc_list


def score_einet(data, K=3, R=3, lr=0.07):
    einet_scores_list = list()

    k_fold = KFold(n_splits=K_FOLDS, random_state=SEED, shuffle=True)

    for i, (train_index, test_index) in enumerate(k_fold.split(data)):
        X_train, X_test = data[train_index], data[test_index]
        X_train = torch.tensor(X_train).float().to(device)
        X_test = torch.tensor(X_test).float().to(device)

        einet_scores = score_one_einet(X_train, X_test, D=1, K=K, R=R, num_classes=3,
                                       class_idx=4, dropout=0.0, epochs=EPOCHS, lr=lr)
        einet_scores_list.append(einet_scores)

    einet_scores_array = np.array(einet_scores_list)

    return einet_scores_array.mean()


def score_cceinet(data, K=3, R=3, lr=0.07):
    cceinet_scores_list = list()

    k_fold = KFold(n_splits=K_FOLDS, random_state=SEED, shuffle=True)

    for i, (train_index, test_index) in enumerate(k_fold.split(data)):
        X_train, X_test = data[train_index], data[test_index]
        X_train = torch.tensor(X_train).float().to(device)
        X_test = torch.tensor(X_test).float().to(device)

        cceinet_scores = score_one_cceinet(
            X_train, X_test, D=1, K=K, R=R, num_classes=3, class_idx=4, dropout=0.0, epochs=EPOCHS, lr=lr)

        cceinet_scores_list.append(cceinet_scores)

    cceinet_scores_array = np.array(cceinet_scores_list)

    return cceinet_scores_array.mean()


def score_einet_proxy(trial):
    K = trial.suggest_int("K", 1, 20)
    R = trial.suggest_int("R", 1, 20)
    lr = trial.suggest_float("lr", 0.0001, 1)

    return score_einet(DATA, K, R, lr)


def score_cceinet_proxy(trial):
    K = trial.suggest_int("K", 1, 20)
    R = trial.suggest_int("R", 1, 20)
    lr = trial.suggest_float("lr", 0.0001, 1)

    return score_cceinet(DATA, K, R, lr)


if __name__ == "__main__":
    einet_study = optuna.create_study(direction="maximize")
    einet_study.optimize(score_einet_proxy, n_trials=TRIALS)

    cceinet_study = optuna.create_study(direction="maximize")
    cceinet_study.optimize(score_cceinet_proxy, n_trials=TRIALS)

    print("einet best params:", einet_study.best_params)
    print("einet best value:", einet_study.best_value)

    print("cceinet best params:", cceinet_study.best_params)
    print("cceinet best value:", cceinet_study.best_value)

    # einet best params: {'K': 15, 'R': 19, 'lr': 0.05349491294870641}
    # einet best value: 94.5066664276123
    # cceinet best params: {'K': 18, 'R': 10, 'lr': 0.522016997420737}
    # cceinet best value: 93.92666650390625
