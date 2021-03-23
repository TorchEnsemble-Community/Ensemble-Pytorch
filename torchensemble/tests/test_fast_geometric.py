import torch
import pytest
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torchensemble import FastGeometricClassifier as clf
from torchensemble import FastGeometricRegressor as reg
from torchensemble.utils.logging import set_logger


set_logger("pytest_fast_geometric")


# Testing data
X_test = torch.Tensor(np.array(([0.5, 0.5], [0.6, 0.6])))

y_test_clf = torch.LongTensor(np.array(([1, 0])))
y_test_reg = torch.FloatTensor(np.array(([0.5, 0.6])))
y_test_reg = y_test_reg.view(-1, 1)


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


def test_fast_geometric_workflow_clf():
    """
    This unit test checks the error message when calling `predict` before
    calling `ensemble`.
    """
    model = clf(estimator=MLP_clf, n_estimators=2, cuda=False)

    model.set_optimizer("Adam")

    # Prepare data
    test = TensorDataset(X_test, y_test_clf)
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    # Training
    with pytest.raises(RuntimeError) as excinfo:
        model.predict(test_loader)
    assert "Please call the `ensemble` method to build" in str(excinfo.value)


def test_fast_geometric_workflow_reg():
    """
    This unit test checks the error message when calling `predict` before
    calling `ensemble`.
    """
    model = reg(estimator=MLP_reg, n_estimators=2, cuda=False)

    model.set_optimizer("Adam")

    # Prepare data
    test = TensorDataset(X_test, y_test_reg)
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    # Training
    with pytest.raises(RuntimeError) as excinfo:
        model.predict(test_loader)
    assert "Please call the `ensemble` method to build" in str(excinfo.value)
