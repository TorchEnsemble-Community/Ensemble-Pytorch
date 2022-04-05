import torch
import pytest
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torchensemble
from torchensemble.utils.logging import set_logger


parallel = [
    torchensemble.FusionClassifier,
    torchensemble.VotingClassifier,
    torchensemble.BaggingClassifier,
    torchensemble.SoftGradientBoostingClassifier,
]

set_logger("pytest_training_params")


# Base estimator
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, X):
        X = X.view(X.size()[0], -1)
        output = self.linear1(X)
        output = self.linear2(output)
        return output


# Trainining data
X_train = torch.Tensor(np.array(([1, 1], [2, 2], [3, 3], [4, 4])))

y_train = torch.LongTensor(np.array(([0, 0, 1, 1])))


# Prepare data
train = TensorDataset(X_train, y_train)
train_loader = DataLoader(train, batch_size=2)


@pytest.mark.parametrize("method", parallel)
def test_parallel(method):
    model = method(estimator=MLP, n_estimators=2, cuda=False)

    # Epochs
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epochs=-1)
    assert "number of training epochs" in str(excinfo.value)

    # Log interval
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, log_interval=-1)
    assert "number of batches to wait" in str(excinfo.value)


def test_gradient_boosting():
    model = torchensemble.GradientBoostingClassifier(
        estimator=MLP, n_estimators=2, cuda=False
    )

    # Epochs
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epochs=-1)
    assert "number of training epochs" in str(excinfo.value)

    # Log interval
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, log_interval=-1)
    assert "number of batches to wait" in str(excinfo.value)

    # Early stoppping round
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, early_stopping_rounds=0)
    assert "number of tolerant rounds" in str(excinfo.value)

    # Shrinkage rate
    model = torchensemble.GradientBoostingClassifier(
        estimator=MLP, n_estimators=2, shrinkage_rate=2, cuda=False
    )
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader)
    assert "shrinkage rate should be in the range" in str(excinfo.value)


def test_snapshot_ensemble():
    model = torchensemble.SnapshotEnsembleClassifier(
        estimator=MLP, n_estimators=2, cuda=False
    )

    # Learning rate clip
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, lr_clip=1)
    assert "lr_clip should be a list or tuple" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, lr_clip=[0, 1, 2])
    assert "lr_clip should only have two elements" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, lr_clip=[1, 0])
    assert "should be smaller than the second" in str(excinfo.value)

    # Epochs
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epochs=-1)
    assert "number of training epochs" in str(excinfo.value)

    # Log interval
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, log_interval=-1)
    assert "number of batches to wait" in str(excinfo.value)

    # Division
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epochs=5)
    assert "should be a multiple of n_estimators" in str(excinfo.value)


def test_adversarial_training():
    model = torchensemble.AdversarialTrainingClassifier(
        estimator=MLP, n_estimators=2, cuda=False
    )

    # Epochs
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epochs=-1)
    assert "number of training epochs" in str(excinfo.value)

    # Epsilon
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epsilon=2)
    assert "step used to generate adversarial samples" in str(excinfo.value)

    # Log interval
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, log_interval=-1)
    assert "number of batches to wait" in str(excinfo.value)


def test_neural_forest():
    model = torchensemble.NeuralForestClassifier(n_estimators=2, depth=-1)

    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epochs=1)
    assert "tree depth should be strictly positive" in str(excinfo.value)

    model = torchensemble.NeuralForestClassifier(
        n_estimators=2,
        depth=3,
        lamda=-1e-3,
    )

    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epochs=1)
    assert "coefficient of the regularization term" in str(excinfo.value)
