import torch
import pytest
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torchensemble
from torchensemble.utils.logging import set_logger


all_clf = [torchensemble.FusionClassifier,
           torchensemble.VotingClassifier,
           torchensemble.BaggingClassifier,
           torchensemble.GradientBoostingClassifier,
           torchensemble.SnapshotEnsembleClassifier,
           torchensemble.AdversarialTrainingClassifier]


all_reg = [torchensemble.FusionRegressor,
           torchensemble.VotingRegressor,
           torchensemble.BaggingRegressor,
           torchensemble.GradientBoostingRegressor,
           torchensemble.SnapshotEnsembleRegressor,
           torchensemble.AdversarialTrainingRegressor]


set_logger("pytest_all_models")


# Base estimator
class MLP_clf(nn.Module):
    def __init__(self):
        super(MLP_clf, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, X):
        X = X.view(X.size()[0], -1)
        output = self.linear1(X)
        output = self.linear2(output)
        return output


class MLP_reg(nn.Module):
    def __init__(self):
        super(MLP_reg, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, X):
        X = X.view(X.size()[0], -1)
        output = self.linear1(X)
        output = self.linear2(output)
        return output


# Trainining data
X_train = torch.Tensor(np.array(([0.1, 0.1],
                                 [0.2, 0.2],
                                 [0.3, 0.3],
                                 [0.4, 0.4])))

y_train_clf = torch.LongTensor(np.array(([0, 0, 1, 1])))
y_train_reg = torch.FloatTensor(np.array(([0.1, 0.2, 0.3, 0.4])))
y_train_reg = y_train_reg.view(-1, 1)

# Testing data
X_test = torch.Tensor(np.array(([0.5, 0.5],
                                [0.6, 0.6],
                                [0.7, 0.7],
                                [0.8, 0.8])))

y_test_clf = torch.LongTensor(np.array(([1, 1, 0, 0])))
y_test_reg = torch.FloatTensor(np.array(([0.5, 0.6, 0.7, 0.8])))
y_test_reg = y_test_reg.view(-1, 1)


@pytest.mark.parametrize("clf", all_clf)
def test_clf(clf):
    """
    This unit test checks the training and evaluating stage of all classifiers.
    """
    epochs = 1
    n_estimators = 2

    model = clf(estimator=MLP_clf, n_estimators=n_estimators, cuda=False)

    # Optimizer
    model.set_optimizer("Adam", lr=1e-3, weight_decay=5e-4)

    # Scheduler (Snapshot Ensemble Excluded)
    if not isinstance(model, torchensemble.SnapshotEnsembleClassifier):
        model.set_scheduler("MultiStepLR", milestones=[2, 4])

    # Prepare data
    train = TensorDataset(X_train, y_train_clf)
    train_loader = DataLoader(train, batch_size=2)
    test = TensorDataset(X_test, y_test_clf)
    test_loader = DataLoader(test, batch_size=2)

    # Snapshot ensemble needs more epochs
    if isinstance(model, torchensemble.SnapshotEnsembleClassifier):
        epochs = 6

    # Train
    model.fit(train_loader,
              epochs=epochs,
              test_loader=test_loader,
              save_model=True)

    # Test
    model.predict(test_loader)


@pytest.mark.parametrize("reg", all_reg)
def test_reg(reg):
    """
    This unit test checks the training and evaluating stage of all regressors.
    """
    epochs = 1
    n_estimators = 2

    model = reg(estimator=MLP_reg, n_estimators=n_estimators, cuda=False)

    # Optimizer
    model.set_optimizer("Adam", lr=1e-3, weight_decay=5e-4)

    # Scheduler (Snapshot Ensemble Excluded)
    if not isinstance(model, torchensemble.SnapshotEnsembleRegressor):
        model.set_scheduler("MultiStepLR", milestones=[2, 4])

    # Prepare data
    train = TensorDataset(X_train, y_train_reg)
    train_loader = DataLoader(train, batch_size=2)
    test = TensorDataset(X_test, y_test_reg)
    test_loader = DataLoader(test, batch_size=2)

    # Snapshot ensemble needs more epochs
    if isinstance(model, torchensemble.SnapshotEnsembleRegressor):
        epochs = 6

    # Train
    model.fit(train_loader,
              epochs=epochs,
              test_loader=test_loader,
              save_model=True)

    # Test
    model.predict(test_loader)


@pytest.mark.parametrize("method", all_clf + all_reg)
def test_estimator_check(method):
    """
    This unit test checks that the input argument estimator should not be
    an instance of a class inherited from nn.Module.
    """
    with pytest.raises(RuntimeError) as excinfo:
        model = method(estimator=MLP_clf(), n_estimators=2, cuda=False)  # noqa: F841,E501
    assert "input argument `estimator` should be a class" in str(excinfo.value)
