import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch

from simple_einet.distributions import CCRatNormal
from simple_einet.einet import CCLEinet, EinetConfig

import numpy as np

parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=14,
    metavar="N",
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--lr", type=float, default=1, metavar="LR", help="learning rate (default: 1)"
)
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--device",
    default="cuda",
    help="Device flag. Can be either 'cpu' or 'cuda'.",
)
parser.add_argument("-K", type=int, default=3)
parser.add_argument("-D", type=int, default=1)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("-R", type=int, default=3)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--class_idx", type=int, default=4)

args = parser.parse_args()

device = torch.device("cpu")
torch.manual_seed(args.seed)

config = EinetConfig(
    num_features=4,
    num_channels=args.D,
    num_sums=args.K,
    num_leaves=args.K,
    num_repetitions=args.R,
    num_classes=args.num_classes,
    leaf_type=CCRatNormal,
    depth=args.depth,
    leaf_kwargs={},
    dropout=0.0)
model = CCLEinet(config, class_idx=args.class_idx).to(device)
print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

iris = datasets.load_iris()
data = np.hstack((iris.data, np.expand_dims(iris.target, axis=1)))
X_train, X_test = train_test_split(data, test_size=0.33, random_state=args.seed)
X_train = torch.tensor(X_train).float().to(device)
X_test = torch.tensor(X_test).float().to(device)


train_conditional = False
test_conditional = False

def accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X, return_conditional=test_conditional)
        predictions = outputs.argmax(-1)
        correct = (y == predictions).sum()
        total = y.shape[0]
        acc = correct / total * 100
        return acc

model.train()
for epoch in range(args.epochs):
    optimizer.zero_grad()

    outputs = model(X_train, return_conditional=train_conditional)
    loss = -outputs.mean()

    loss.backward()
    optimizer.step()

    model.eval()
    acc_train = accuracy(model, X_train, X_train[:, args.class_idx])
    acc_test = accuracy(model, X_test, X_test[:, args.class_idx])
    model.train()

    print(
        "Train Epoch: {}\tLoss: {:3.2f}\t\tAccuracy Train: {:2.2f} %\t\tAccuracy Test: {:2.2f} %".format(
            epoch,
            loss.item(),
            acc_train,
            acc_test,
        )
    )

