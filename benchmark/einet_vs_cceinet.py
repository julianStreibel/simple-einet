from sklearn import datasets
import torch

from simple_einet.distributions import CCRatNormal, RatNormal
from simple_einet.einet import Einet, CCLEinet, EinetConfig

import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


SEED = 1

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


def score_cceinet(train_data, test_data, D=1, K=3, R=3, num_classes=3, class_idx=4, dropout=0.0, epochs=40, lr=0.2):
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
    print("Number of parameters in CCEinet:", sum(p.numel()
                                                  for p in model.parameters() if p.requires_grad))
    print("Number of parameters in leaves CCEinet", sum(p.numel()
                                                        for p in model.leaf.parameters() if p.requires_grad))
    print()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    loss_list = list()
    train_acc_list = list()
    test_acc_list = list()

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train, y_train)
        loss = -outputs.mean()

        loss.backward()
        optimizer.step()

        model.eval()
        acc_train = accuracy(model, X_train, y_train)
        acc_test = accuracy(model, X_test, y_test)
        model.train()

        loss_list.append(loss.item())
        train_acc_list.append(acc_train.item())
        test_acc_list.append(acc_test.item())

    return loss_list, train_acc_list, test_acc_list


def score_einet(train_data, test_data, D=1, K=3, R=3, num_classes=3, class_idx=4, dropout=0.0, epochs=40, lr=0.07):

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
    print("Number of parameters in Einet:", sum(p.numel()
                                                for p in model.parameters() if p.requires_grad))
    print("Number of parameters in leaves Einet", sum(p.numel()
                                                      for p in model.leaf.parameters() if p.requires_grad))
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_list = list()
    train_acc_list = list()
    test_acc_list = list()

    cross_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = cross_entropy(outputs, y_train)
        # loss = -outputs.mean()
        loss.backward()
        optimizer.step()

        acc_train = accuracy(model, X_train, y_train)
        acc_test = accuracy(model, X_test, y_test)

        loss_list.append(loss.item())
        train_acc_list.append(acc_train.item())
        test_acc_list.append(acc_test.item())

    return loss_list, train_acc_list, test_acc_list


def k_fold(
    K=3,
    R=3,
    LR_EINET=0.07,
    LR_CCEINET=0.2,
    EPOCHS=40,
    K_FOLDS=10,
    show=False
):
    iris = datasets.load_iris()
    data = np.hstack((iris.data, np.expand_dims(iris.target, axis=1)))

    einet_scores_list = list()
    cceinet_scores_list = list()

    k_fold = KFold(n_splits=K_FOLDS, random_state=SEED, shuffle=True)

    for i, (train_index, test_index) in enumerate(k_fold.split(data)):
        X_train, X_test = data[train_index], data[test_index]
        X_train = torch.tensor(X_train).float().to(device)
        X_test = torch.tensor(X_test).float().to(device)

        einet_scores = score_einet(X_train, X_test, D=1, K=K, R=R, num_classes=3,
                                   class_idx=4, dropout=0.0, epochs=EPOCHS, lr=LR_EINET)
        cceinet_scores = score_cceinet(
            X_train, X_test, D=1, K=K, R=R, num_classes=3, class_idx=4, dropout=0.0, epochs=EPOCHS, lr=LR_CCEINET)

        einet_scores_list.append(einet_scores)
        cceinet_scores_list.append(cceinet_scores)

    einet_scores_array = np.array(einet_scores_list)
    cceinet_scores_array = np.array(cceinet_scores_list)

    einet_scores_mean = einet_scores_array.mean(axis=0)
    einet_scores_std = einet_scores_array.std(axis=0)

    cceinet_scores_mean = cceinet_scores_array.mean(axis=0)
    cceinet_scores_std = cceinet_scores_array.std(axis=0)

    x = np.arange(einet_scores_mean.shape[1])

    plt.figure()
    for i, (name, color) in enumerate([("loss", "blue"), ("train acc", "red"), ("test acc", "green")]):
        plt.plot(x, einet_scores_mean[i],
                 label=f"einet {name} mean", color=f"{color}")
        plt.fill_between(
            x,
            einet_scores_mean[i] - einet_scores_std[i],
            einet_scores_mean[i] + einet_scores_std[i],
            alpha=0.2,
            color=color
        )
    plt.title(f"Einet: K={K}, R={R}, lr={LR_EINET}")
    plt.legend()

    plt.figure()
    for i, (name, color) in enumerate([("loss", "blue"), ("train acc", "red"), ("test acc", "green")]):
        plt.plot(x, cceinet_scores_mean[i],
                 label=f"cceinet {name} mean", color=f"dark{color}")
        plt.fill_between(
            x,
            cceinet_scores_mean[i] - cceinet_scores_std[i],
            cceinet_scores_mean[i] + cceinet_scores_std[i],
            alpha=0.2,
            color=color
        )
    plt.title(f"CCEinet: K={K}, R={R}, lr={LR_CCEINET}")
    plt.legend()
    if show:
        plt.show()


if __name__ == "__main__":
    # lr_cceinet = np.linspace(0.01, 1, 20)
    # lr_einet = np.linspace(0.001, 0.1, 20)
    # for lr_cc, lr_ei in zip(lr_cceinet, lr_einet):
    #     k_fold(LR_EINET=lr_ei, LR_CCEINET=lr_cc, EPOCHS=100)
    # plt.show()

    # best params for einet
    k_fold(LR_EINET=0.05349491294870641, LR_CCEINET=0.05349491294870641,
           EPOCHS=100, K=15, R=19, K_FOLDS=5)

    # best parrams for cceinet
    k_fold(LR_EINET=0.522016997420737, LR_CCEINET=0.522016997420737,
           EPOCHS=100, K=18, R=10, K_FOLDS=5)
    plt.show()
