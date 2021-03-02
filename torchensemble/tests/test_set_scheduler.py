import torch
import pytest
import torchensemble
import torch.nn as nn


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


def test_set_scheduler_LambdaLR():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())
    lr_lambda = lambda x: x * 0.1  # noqa: E731
    torchensemble.utils.set_module.set_scheduler(
        optimizer, "LambdaLR", lr_lambda=lr_lambda
    )


def test_set_scheduler_MultiplicativeLR():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())
    lr_lambda = lambda x: x * 0.1  # noqa: E731
    torchensemble.utils.set_module.set_scheduler(
        optimizer, "MultiplicativeLR", lr_lambda=lr_lambda
    )


def test_set_scheduler_StepLR():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())
    torchensemble.utils.set_module.set_scheduler(
        optimizer, "StepLR", step_size=50
    )


def test_set_scheduler_MultiStepLR():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())
    torchensemble.utils.set_module.set_scheduler(
        optimizer, "MultiStepLR", milestones=[50, 100]
    )


def test_set_scheduler_ExponentialLR():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())
    torchensemble.utils.set_module.set_scheduler(
        optimizer, "ExponentialLR", gamma=0.1
    )


def test_set_scheduler_CosineAnnealingLR():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())
    torchensemble.utils.set_module.set_scheduler(
        optimizer, "CosineAnnealingLR", T_max=100
    )


def test_set_scheduler_ReduceLROnPlateau():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())
    torchensemble.utils.set_module.set_scheduler(
        optimizer, "ReduceLROnPlateau"
    )


def test_set_scheduler_Unknown():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())

    with pytest.raises(NotImplementedError) as excinfo:
        torchensemble.utils.set_module.set_scheduler(optimizer, "Unknown")
    assert "Unrecognized scheduler" in str(excinfo.value)
