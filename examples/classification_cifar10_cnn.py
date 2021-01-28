"""Example on classification using CIFAR-10."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier
from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier

from torchensemble.utils.logging import set_logger


def display_records(records, logger):
    msg = (
        "{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, acc in records:
        logger.info(msg.format(method, acc, training_time, evaluating_time))


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    # Hyper-parameters
    n_estimators = 10
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 100

    # Utils
    batch_size = 128
    data_dir = "../../Dataset/cifar"  # MODIFY THIS IF YOU WANT
    records = []
    torch.manual_seed(0)

    # Load data
    train_transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_loader = DataLoader(
        datasets.CIFAR10(data_dir, train=True, download=True,
                         transform=train_transformer),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.CIFAR10(data_dir, train=False, transform=test_transformer),
        batch_size=batch_size,
        shuffle=True,
    )

    logger = set_logger("classification_cifar10_cnn")

    # FusionClassifier
    model = FusionClassifier(
        estimator=LeNet5,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    # Evaluating
    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("FusionClassifier", training_time, evaluating_time,
                    testing_acc))

    # VotingClassifier
    model = VotingClassifier(
        estimator=LeNet5,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("VotingClassifier", training_time, evaluating_time,
                    testing_acc))

    # BaggingClassifier
    model = BaggingClassifier(
        estimator=LeNet5,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("BaggingClassifier", training_time, evaluating_time,
                    testing_acc))

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(
        estimator=LeNet5,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("GradientBoostingClassifier", training_time,
                    evaluating_time, testing_acc))

    # SnapshotEnsembleClassifier
    model = SnapshotEnsembleClassifier(
        estimator=LeNet5,
        n_estimators=n_estimators,
        cuda=True
    )

    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("SnapshotEnsembleClassifier", training_time,
                    evaluating_time, testing_acc))

    # Print results on different ensemble methods
    display_records(records, logger)
