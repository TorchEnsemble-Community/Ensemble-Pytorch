"""Example on classification using CIFAR-10."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchensemble.utils import get_default_logger
from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier


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

        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):

        # CONV layers
        output = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        output = F.max_pool2d(F.relu(self.conv2(output)), (2, 2))
        output = output.view(-1, self.num_flat_features(output))

        # FC layers
        output = F.relu(self.fc1(output))
        output = F.dropout(output)
        output = F.relu(self.fc2(output))
        output = F.dropout(output)
        output = self.fc3(output)

        return output

    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


if __name__ == "__main__":

    # Hyper-parameters
    n_estimators = 1
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 1

    # Utils
    batch_size = 128
    data_dir = "../../Dataset/cifar"  # MODIFY THIS IF YOU WANT
    records = []
    torch.manual_seed(0)

    # Load data
    transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_loader = DataLoader(
        datasets.CIFAR10(data_dir, train=True, download=True,
                         transform=transformer),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.CIFAR10(
            data_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    logger = get_default_logger("INFO", "classification_cifar10_cnn", "DEBUG")

    # FusionClassifier
    model = FusionClassifier(
        estimator=LeNet5,
        n_estimators=n_estimators,
        cuda=True,
        n_jobs=1,
        logger=logger
    )

    tic = time.time()
    model.fit(train_loader, lr, weight_decay, epochs, "Adam")
    toc = time.time()
    training_time = toc - tic

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
        cuda=True,
        n_jobs=1,
        logger=logger
    )

    tic = time.time()
    model.fit(train_loader, lr, weight_decay, epochs, "Adam")
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
        cuda=True,
        n_jobs=1,
        logger=logger
    )

    tic = time.time()
    model.fit(train_loader, lr, weight_decay, epochs, "Adam")
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
        cuda=True,
        logger=logger
    )

    tic = time.time()
    model.fit(train_loader, lr, weight_decay, epochs, "Adam")
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(("GradientBoostingClassifier", training_time,
                    evaluating_time, testing_acc))

    # Print results on different ensemble methods
    display_records(records, logger)
