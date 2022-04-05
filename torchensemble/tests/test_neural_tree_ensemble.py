import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from torchensemble.utils import io
from torchensemble.utils.logging import set_logger
from torchensemble import NeuralForestClassifier, NeuralForestRegressor


np.random.seed(0)
torch.manual_seed(0)
set_logger("pytest_neural_tree_ensemble")


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


def test_neural_forest_classifier():
    """
    This unit test checks the training and evaluating stage of
    NeuralForestClassifier.
    """
    epochs = 1
    n_estimators = 2
    depth = 3
    lamda = 1e-3

    model = NeuralForestClassifier(
        n_estimators=n_estimators,
        depth=depth,
        lamda=lamda,
        cuda=False,
        n_jobs=1,
    )

    model.set_optimizer("Adam", lr=1e-4, weight_decay=5e-4)

    # Prepare data
    train = TensorDataset(X_train, y_train_clf)
    train_loader = DataLoader(train, batch_size=2, shuffle=False)
    test = TensorDataset(X_test, y_test_clf)
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    # Train
    model.fit(train_loader, epochs=epochs, test_loader=test_loader)

    # Evaluate
    model.evaluate(test_loader)

    # Predict
    for _, (data, target) in enumerate(test_loader):
        model.predict(data)
        break

    # Reload
    new_model = NeuralForestClassifier(
        n_estimators=n_estimators,
        depth=depth,
        lamda=lamda,
        cuda=False,
        n_jobs=1,
    )
    io.load(new_model)

    new_model.evaluate(test_loader)

    for _, (data, target) in enumerate(test_loader):
        new_model.predict(data)
        break


def test_neural_forest_regressor():
    """
    This unit test checks the training and evaluating stage of
    NeuralForestClassifier.
    """
    epochs = 1
    n_estimators = 2
    depth = 3
    lamda = 1e-3

    model = NeuralForestRegressor(
        n_estimators=n_estimators,
        depth=depth,
        lamda=lamda,
        cuda=False,
        n_jobs=1,
    )

    model.set_optimizer("Adam", lr=1e-4, weight_decay=5e-4)

    # Prepare data
    train = TensorDataset(X_train, y_train_reg)
    train_loader = DataLoader(train, batch_size=2, shuffle=False)
    test = TensorDataset(X_test, y_test_reg)
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    # Train
    model.fit(train_loader, epochs=epochs, test_loader=test_loader)

    # Evaluate
    model.evaluate(test_loader)

    # Predict
    for _, (data, target) in enumerate(test_loader):
        model.predict(data)
        break

    # Reload
    new_model = NeuralForestRegressor(
        n_estimators=n_estimators,
        depth=depth,
        lamda=lamda,
        cuda=False,
        n_jobs=1,
    )
    io.load(new_model)

    new_model.evaluate(test_loader)

    for _, (data, target) in enumerate(test_loader):
        new_model.predict(data)
        break
