import torch
import pytest
import torchensemble
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# All regressors
all_reg = [
    torchensemble.FusionRegressor,
    torchensemble.VotingRegressor,
    torchensemble.BaggingRegressor,
    torchensemble.GradientBoostingRegressor,
    torchensemble.AdversarialTrainingRegressor,
    torchensemble.FastGeometricRegressor,
    torchensemble.SoftGradientBoostingRegressor,
]

# All classifiers
all_clf = [
    torchensemble.FusionClassifier,
    torchensemble.VotingClassifier,
    torchensemble.BaggingClassifier,
    torchensemble.GradientBoostingClassifier,
    torchensemble.AdversarialTrainingClassifier,
    torchensemble.FastGeometricClassifier,
    torchensemble.SoftGradientBoostingClassifier,
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


# Training data
X_train = torch.rand(2, 2)
y_train_reg = torch.rand(2, 2)
X_train_clf = torch.Tensor(
    np.array(([0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]))
)
y_train_clf = torch.LongTensor(np.array(([0, 0, 1, 1])))

# Testing data
X_test = torch.rand(2, 2)
y_test_reg = torch.rand(2, 2)
numpy_X_test = np.array(([0.5, 0.5], [0.6, 0.6]))
X_test_clf = torch.Tensor(numpy_X_test)
y_test_clf = torch.LongTensor(np.array(([1, 0])))

train_data_reg = TensorDataset(X_train, y_train_reg)
train_loader_reg = DataLoader(train_data_reg, batch_size=5, shuffle=True)
test_data_reg = TensorDataset(X_test, y_test_reg)
test_loader_reg = DataLoader(test_data_reg, batch_size=5, shuffle=True)

train_data_clf = TensorDataset(X_train_clf, y_train_clf)
train_loader_clf = DataLoader(train_data_clf, batch_size=5, shuffle=True)
test_data_clf = TensorDataset(X_test_clf, y_test_clf)
test_loader_clf = DataLoader(test_data_clf, batch_size=5, shuffle=True)


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
    "test_dataloader", [test_loader_reg, None]
)  # ReduceLROnPlateau scheduling should work when there is no test data
@pytest.mark.parametrize(
    "n_estimators", [1, 5]
)  # LR scheduling works for 1 as well as many estimators
@pytest.mark.parametrize("ensemble_model", all_reg)
def test_fit_w_all_schedulers(
    scheduler_dict, test_dataloader, n_estimators, ensemble_model
):
    """Test if LR schedulers work when `fit` is called."""
    model = ensemble_model(
        estimator=MLP, n_estimators=n_estimators, cuda=False
    )
    model.set_optimizer("Adam", lr=1e-1)
    model.set_scheduler(**scheduler_dict)
    model.fit(
        train_loader_reg,
        epochs=50,
        test_loader=test_dataloader,
        save_model=False,
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
    "test_dataloader", [test_loader_clf, None]
)  # ReduceLROnPlateau scheduling should work when there is no test data
@pytest.mark.parametrize(
    "n_estimators", [1, 5]
)  # LR scheduling works for 1 as well as many estimators
@pytest.mark.parametrize("ensemble_model", all_clf)
def test_fit_w_all_schedulers_clf(
    scheduler_dict, test_dataloader, n_estimators, ensemble_model
):
    """Test if LR schedulers work when `fit` is called."""
    model = ensemble_model(
        estimator=MLP_clf, n_estimators=n_estimators, cuda=False
    )
    model.set_optimizer("Adam", lr=1e-1)
    model.set_scheduler(**scheduler_dict)
    model.fit(
        train_loader_clf,
        epochs=50,
        test_loader=test_dataloader,
        save_model=False,
    )


def test_set_scheduler_Unknown():
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters())

    with pytest.raises(NotImplementedError) as excinfo:
        torchensemble.utils.set_module.set_scheduler(optimizer, "Unknown")
    assert "Unrecognized scheduler" in str(excinfo.value)
