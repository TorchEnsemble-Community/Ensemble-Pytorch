"""Example on regression using YearPredictionMSD."""

import time
import torch
import numbers
import torch.nn as nn
from torch.nn import functional as F
from sklearn.preprocessing import scale
from sklearn.datasets import load_svmlight_file
from torch.utils.data import TensorDataset, DataLoader

from torchensemble.fusion import FusionRegressor
from torchensemble.voting import VotingRegressor
from torchensemble.bagging import BaggingRegressor
from torchensemble.gradient_boosting import GradientBoostingRegressor
from torchensemble.snapshot_ensemble import SnapshotEnsembleRegressor

from torchensemble.utils.logging import set_logger


def load_data(batch_size):

    # The dataset can be downloaded from:
    #   https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#YearPredictionMSD

    if not isinstance(batch_size, numbers.Integral):
        msg = "`batch_size` should be an integer, but got {} instead."
        raise ValueError(msg.format(batch_size))

    # MODIFY THE PATH IF YOU WANT
    train_path = "../../Dataset/LIBSVM/yearpredictionmsd_training"
    test_path = "../../Dataset/LIBSVM/yearpredictionmsd_testing"

    train = load_svmlight_file(train_path)
    test = load_svmlight_file(test_path)

    # Numpy array -> Tensor
    X_train, X_test = (
        torch.FloatTensor(train[0].toarray()),
        torch.FloatTensor(test[0].toarray()),
    )

    y_train, y_test = (
        torch.FloatTensor(scale(train[1]).reshape(-1, 1)),
        torch.FloatTensor(scale(test[1]).reshape(-1, 1)),
    )

    # Tensor -> Data loader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def display_records(records, logger):
    msg = (
        "{:<28} | Testing MSE: {:.2f} | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, mse in records:
        logger.info(msg.format(method, mse, training_time, evaluating_time))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(90, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


if __name__ == "__main__":

    # Hyper-parameters
    n_estimators = 10
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 50

    # Utils
    batch_size = 512
    records = []
    torch.manual_seed(0)

    # Load data
    train_loader, test_loader = load_data(batch_size)
    print("Finish loading data...\n")

    logger = set_logger("regression_YearPredictionMSD_mlp")

    # FusionRegressor
    model = FusionRegressor(
        estimator=MLP,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("FusionRegressor", training_time, evaluating_time,
                    testing_mse))

    # VotingRegressor
    model = VotingRegressor(
        estimator=MLP,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("VotingRegressor", training_time, evaluating_time,
                    testing_mse))

    # BaggingRegressor
    model = BaggingRegressor(
        estimator=MLP,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("BaggingRegressor", training_time, evaluating_time,
                    testing_mse))

    # GradientBoostingRegressor
    model = GradientBoostingRegressor(
        estimator=MLP,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("GradientBoostingRegressor", training_time,
                    evaluating_time, testing_mse))

    # SnapshotEnsembleRegressor
    model = SnapshotEnsembleRegressor(
        estimator=MLP,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("SnapshotEnsembleRegressor", training_time,
                    evaluating_time, testing_acc))

    # Print results on different ensemble methods
    display_records(records, logger)
