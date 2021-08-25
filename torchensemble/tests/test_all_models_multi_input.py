import torch
import pytest
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torchensemble
from torchensemble.utils import io
from torchensemble.utils.logging import set_logger


# All classifiers
all_clf = [
    torchensemble.FusionClassifier,
    torchensemble.VotingClassifier,
    torchensemble.BaggingClassifier,
    torchensemble.GradientBoostingClassifier,
    torchensemble.SnapshotEnsembleClassifier,
    torchensemble.AdversarialTrainingClassifier,
    torchensemble.FastGeometricClassifier,
    torchensemble.SoftGradientBoostingClassifier,
]


# All regressors
all_reg = [
    torchensemble.FusionRegressor,
    torchensemble.VotingRegressor,
    torchensemble.BaggingRegressor,
    torchensemble.GradientBoostingRegressor,
    torchensemble.SnapshotEnsembleRegressor,
    torchensemble.AdversarialTrainingRegressor,
    torchensemble.FastGeometricRegressor,
    torchensemble.SoftGradientBoostingRegressor,
]


np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cpu")
logger = set_logger("pytest_all_models_multiple_input")


# Base estimator
class MLP_clf(nn.Module):
    def __init__(self):
        super(MLP_clf, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, X_1, X_2):
        X_1 = X_1.view(X_1.size()[0], -1)
        X_2 = X_2.view(X_2.size()[0], -1)
        output_1 = self.linear1(X_1)
        output_1 = self.linear2(output_1)
        output_2 = self.linear1(X_2)
        output_2 = self.linear2(output_2)
        return 0.5 * output_1 + 0.5 * output_2


class MLP_reg(nn.Module):
    def __init__(self):
        super(MLP_reg, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, X_1, X_2):
        X_1 = X_1.view(X_1.size()[0], -1)
        X_2 = X_2.view(X_2.size()[0], -1)
        output_1 = self.linear1(X_1)
        output_1 = self.linear2(output_1)
        output_2 = self.linear1(X_2)
        output_2 = self.linear2(output_2)
        return 0.5 * output_1 + 0.5 * output_2


# Training data
X_train = torch.Tensor(
    np.array(([0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]))
)

y_train_clf = torch.LongTensor(np.array(([0, 0, 1, 1])))
y_train_reg = torch.FloatTensor(np.array(([0.1, 0.2, 0.3, 0.4])))
y_train_reg = y_train_reg.view(-1, 1)


# Testing data
numpy_X_test = np.array(([0.5, 0.5], [0.6, 0.6]))
X_test = torch.Tensor(numpy_X_test)

y_test_clf = torch.LongTensor(np.array(([1, 0])))
y_test_reg = torch.FloatTensor(np.array(([0.5, 0.6])))
y_test_reg = y_test_reg.view(-1, 1)


@pytest.mark.parametrize("clf", all_clf)
def test_clf_class(clf):
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

    # Prepare data with multiple inputs
    train = TensorDataset(X_train, X_train, y_train_clf)
    train_loader = DataLoader(train, batch_size=2, shuffle=False)
    test = TensorDataset(X_test, X_test, y_test_clf)
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    # Snapshot Ensemble needs more epochs
    if isinstance(model, torchensemble.SnapshotEnsembleClassifier):
        epochs = 6

    # Train
    model.fit(train_loader, epochs=epochs, test_loader=test_loader)

    # Evaluate
    model.evaluate(test_loader)

    # Predict
    for _, elem in enumerate(test_loader):
        data, target = io.split_data_target(elem, device)
        model.predict(*data)
        break

    # Reload
    new_model = clf(estimator=MLP_clf, n_estimators=n_estimators, cuda=False)
    io.load(new_model)

    new_model.evaluate(test_loader)

    for _, elem in enumerate(test_loader):
        data, target = io.split_data_target(elem, device)
        new_model.predict(*data)
        break


@pytest.mark.parametrize("clf", all_clf)
def test_clf_object(clf):
    """
    This unit test checks the training and evaluating stage of all classifiers.
    """
    epochs = 1
    n_estimators = 2

    model = clf(estimator=MLP_clf(), n_estimators=n_estimators, cuda=False)

    # Optimizer
    model.set_optimizer("Adam", lr=1e-3, weight_decay=5e-4)

    # Scheduler (Snapshot Ensemble Excluded)
    if not isinstance(model, torchensemble.SnapshotEnsembleClassifier):
        model.set_scheduler("MultiStepLR", milestones=[2, 4])

    # Prepare data with multiple inputs
    train = TensorDataset(X_train, X_train, y_train_clf)
    train_loader = DataLoader(train, batch_size=2, shuffle=False)
    test = TensorDataset(X_test, X_test, y_test_clf)
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    # Snapshot Ensemble needs more epochs
    if isinstance(model, torchensemble.SnapshotEnsembleClassifier):
        epochs = 6

    # Train
    model.fit(train_loader, epochs=epochs, test_loader=test_loader)

    # Evaluate
    model.evaluate(test_loader)

    # Predict
    for _, elem in enumerate(test_loader):
        data, target = io.split_data_target(elem, device)
        model.predict(*data)
        break

    # Reload
    new_model = clf(estimator=MLP_clf(), n_estimators=n_estimators, cuda=False)
    io.load(new_model)

    new_model.evaluate(test_loader)

    for _, elem in enumerate(test_loader):
        data, target = io.split_data_target(elem, device)
        new_model.predict(*data)
        break


@pytest.mark.parametrize("reg", all_reg)
def test_reg_class(reg):
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

    # Prepare data with multiple inputs
    train = TensorDataset(X_train, X_train, y_train_reg)
    train_loader = DataLoader(train, batch_size=2, shuffle=False)
    test = TensorDataset(X_test, X_test, y_test_reg)
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    # Snapshot Ensemble needs more epochs
    if isinstance(model, torchensemble.SnapshotEnsembleRegressor):
        epochs = 6

    # Train
    model.fit(train_loader, epochs=epochs, test_loader=test_loader)

    # Evaluate
    model.evaluate(test_loader)

    # Predict
    for _, elem in enumerate(test_loader):
        data, target = io.split_data_target(elem, device)
        model.predict(*data)
        break

    # Reload
    new_model = reg(estimator=MLP_reg, n_estimators=n_estimators, cuda=False)
    io.load(new_model)

    new_model.evaluate(test_loader)

    for _, elem in enumerate(test_loader):
        data, target = io.split_data_target(elem, device)
        new_model.predict(*data)
        break


@pytest.mark.parametrize("reg", all_reg)
def test_reg_object(reg):
    """
    This unit test checks the training and evaluating stage of all regressors.
    """
    epochs = 1
    n_estimators = 2

    model = reg(estimator=MLP_reg(), n_estimators=n_estimators, cuda=False)

    # Optimizer
    model.set_optimizer("Adam", lr=1e-3, weight_decay=5e-4)

    # Scheduler (Snapshot Ensemble Excluded)
    if not isinstance(model, torchensemble.SnapshotEnsembleRegressor):
        model.set_scheduler("MultiStepLR", milestones=[2, 4])

    # Prepare data with multiple inputs
    train = TensorDataset(X_train, X_train, y_train_reg)
    train_loader = DataLoader(train, batch_size=2, shuffle=False)
    test = TensorDataset(X_test, X_test, y_test_reg)
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    # Snapshot Ensemble needs more epochs
    if isinstance(model, torchensemble.SnapshotEnsembleRegressor):
        epochs = 6

    # Train
    model.fit(train_loader, epochs=epochs, test_loader=test_loader)

    # Evaluate
    model.evaluate(test_loader)

    # Predict
    for _, elem in enumerate(test_loader):
        data, target = io.split_data_target(elem, device)
        model.predict(*data)
        break

    # Reload
    new_model = reg(estimator=MLP_reg(), n_estimators=n_estimators, cuda=False)
    io.load(new_model)

    new_model.evaluate(test_loader)

    for _, elem in enumerate(test_loader):
        data, target = io.split_data_target(elem, device)
        new_model.predict(*data)
        break


def test_split_data_target_invalid_data_type():
    with pytest.raises(ValueError) as excinfo:
        io.split_data_target(0.0, device, logger)
    assert "Invalid dataloader" in str(excinfo.value)


def test_split_data_target_invalid_list_length():
    with pytest.raises(ValueError) as excinfo:
        io.split_data_target([0.0], device, logger)
    assert "should at least contain two tensors" in str(excinfo.value)
