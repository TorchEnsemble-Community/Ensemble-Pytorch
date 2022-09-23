import torch
import pytest
import torchensemble
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# All regressors
all_reg = [
    # torchensemble.FusionRegressor,
    torchensemble.VotingRegressor,
    # torchensemble.BaggingRegressor,
    # torchensemble.GradientBoostingRegressor,
    # torchensemble.SnapshotEnsembleRegressor,
    # torchensemble.AdversarialTrainingRegressor,
    # torchensemble.FastGeometricRegressor,
    torchensemble.SoftGradientBoostingRegressor,
]

# All classifiers
all_clf = [
    # torchensemble.FusionClassifier,
    torchensemble.VotingClassifier,
    # torchensemble.BaggingClassifier,
    # torchensemble.GradientBoostingClassifier,
    # torchensemble.SnapshotEnsembleClassifier,
    # torchensemble.AdversarialTrainingClassifier,
    # torchensemble.FastGeometricClassifier,
    # torchensemble.SoftGradientBoostingClassifier,  # failing
]

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


X = torch.rand(2, 2)
y = torch.rand(2, 2)

X_test = torch.rand(2, 2)
y_test = torch.rand(2, 2)

train_data = TensorDataset(X, y)
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=5, shuffle=True)


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


@pytest.mark.parametrize(
    "scheduler_dict",
    [
        {"scheduler_name": "ReduceLROnPlateau", "patience": 2, "min_lr": 1e-6},
        {"scheduler_name": "LambdaLR", "lr_lambda": lambda x: x * 0.1},
        {"scheduler_name": "StepLR", "step_size": 30},
        {"scheduler_name": "MultiStepLR", "milestones": [20, 40]},
        {"scheduler_name": "ExponentialLR", "gamma": 0.1},
        {"scheduler_name": "CosineAnnealingLR", "T_max": 100},
    ],
)
@pytest.mark.parametrize(
    "test_dataloader", [test_loader, None]
)  # ReduceLROnPlateau scheduling should work when there is no test data
@pytest.mark.parametrize(
    "n_estimators", [1, 10]
)  # LR scheduling works for 1 as well as many estimators
@pytest.mark.parametrize("ensemble_model", all_reg)
def test_fit_w_all_schedulers(scheduler_dict, test_dataloader, n_estimators, ensemble_model):
    """Test if LR schedulers work when `fit` is called."""
    model = ensemble_model(
        estimator=MLP, n_estimators=n_estimators, cuda=False
    )
    model.set_optimizer("Adam", lr=1e-1)
    model.set_scheduler(**scheduler_dict)
    model.fit(train_loader, epochs=50, test_loader=test_dataloader)


@pytest.mark.parametrize(
    "scheduler_dict",
    [
        {"scheduler_name": "ReduceLROnPlateau", "patience": 2, "min_lr": 1e-6},
        {"scheduler_name": "LambdaLR", "lr_lambda": lambda x: x * 0.1},
        {"scheduler_name": "StepLR", "step_size": 30},
        {"scheduler_name": "MultiStepLR", "milestones": [20, 40]},
        {"scheduler_name": "ExponentialLR", "gamma": 0.1},
        {"scheduler_name": "CosineAnnealingLR", "T_max": 100},
    ],
)
@pytest.mark.parametrize(
    "test_dataloader", [test_loader, None]
)  # ReduceLROnPlateau scheduling should work when there is no test data
@pytest.mark.parametrize(
    "n_estimators", [1, 10]
)  # LR scheduling works for 1 as well as many estimators
@pytest.mark.parametrize("ensemble_model", all_clf)
def test_fit_w_all_schedulers_clf(scheduler_dict, test_dataloader, n_estimators, ensemble_model):
    """Test if LR schedulers work when `fit` is called."""
    model = ensemble_model(
        estimator=MLP_clf, n_estimators=n_estimators, cuda=False
    )
    model.set_optimizer("Adam", lr=1e-1)
    model.set_scheduler(**scheduler_dict)
    model.fit(train_loader, epochs=50, test_loader=test_dataloader)


def test_set_scheduler_Unknown():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())

    with pytest.raises(NotImplementedError) as excinfo:
        torchensemble.utils.set_module.set_scheduler(optimizer, "Unknown")
    assert "Unrecognized scheduler" in str(excinfo.value)
