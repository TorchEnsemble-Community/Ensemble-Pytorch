import torch
import pytest
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torchensemble
from torchensemble.utils import get_default_logger


parallel = [torchensemble.FusionClassifier,
            torchensemble.VotingClassifier,
            torchensemble.BaggingClassifier]

logger = get_default_logger("INFO", "pytest_params", "DEBUG")

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
X_train = torch.Tensor(np.array(([1, 1],
                                 [2, 2],
                                 [3, 3],
                                 [4, 4])))

y_train = torch.LongTensor(np.array(([0, 0, 1, 1])))


# Prepare data
train = TensorDataset(X_train, y_train)
train_loader = DataLoader(train, batch_size=2)


@pytest.mark.parametrize("method", parallel)
def test_parallel(method):
    model = method(estimator=MLP, n_estimators=2, logger=logger, cuda=False)

    # Learning rate
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, lr=-1)
    assert "learning rate of optimizer" in str(excinfo.value)

    # Weight decay
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, weight_decay=-1)
    assert "weight decay of optimizer" in str(excinfo.value)

    # Epochs
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, epochs=-1)
    assert "number of training epochs" in str(excinfo.value)

    # Log interval
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, log_interval=-1)
    assert "number of batches to wait" in str(excinfo.value)


def test_snapshot_ensemble():
    model = torchensemble.SnapshotEnsembleClassifier(estimator=MLP,
                                                     n_estimators=2,
                                                     logger=logger,
                                                     cuda=False)
    # Learning rate
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, init_lr=-1)
    assert "initial learning rate" in str(excinfo.value)

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

    # Weight decay
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader, weight_decay=-1)
    assert "weight decay of optimizer" in str(excinfo.value)

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
