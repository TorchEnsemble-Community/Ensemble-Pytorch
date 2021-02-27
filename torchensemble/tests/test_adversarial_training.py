import torch
import pytest
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torchensemble
from torchensemble.utils.logging import set_logger


set_logger("pytest_adversarial_training")


X_train = torch.Tensor(np.array(([1, 1], [2, 2], [3, 3], [4, 4])))
y_train_clf = torch.LongTensor(np.array(([0, 0, 1, 1])))


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


def test_adversarial_training_range():
    """
    This unit test checks the input range check in adversarial_training.py.
    """
    model = torchensemble.AdversarialTrainingClassifier(
        estimator=MLP_clf, n_estimators=2, cuda=False
    )

    model.set_optimizer("Adam")

    # Prepare data
    train = TensorDataset(X_train, y_train_clf)
    train_loader = DataLoader(train, batch_size=2)

    # Training
    with pytest.raises(ValueError) as excinfo:
        model.fit(train_loader)
    assert "input range of samples passed to adversarial" in str(excinfo.value)
