import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier
from torchensemble.utils.logging import set_logger


# The class `BasicBlock` and `ResNet` is modified from:
#   https://github.com/kuangliu/pytorch-cifar
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == "__main__":

    # Hyper-parameters
    n_estimators = 5
    lr = 1e-1
    weight_decay = 5e-4
    momentum = 0.9
    epochs = 200  # i.e., 40 epochs for each snapshot

    # Utils
    batch_size = 128
    data_dir = "../../Dataset/cifar"  # MODIFY THIS IF YOU WANT
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    # Load data
    train_transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    test_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    train_loader = DataLoader(
        datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transformer
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.CIFAR10(data_dir, train=False, transform=test_transformer),
        batch_size=batch_size,
        shuffle=True,
    )

    # Set the Logger
    logger, tb_logger = set_logger("snapshot_ensemble_cifar10_resnet18", use_tb_logger=True)

    # Choose the Ensemble Method
    model = SnapshotEnsembleClassifier(
        estimator=ResNet,
        estimator_args={"block": BasicBlock, "num_blocks": [2, 2, 2, 2]},
        n_estimators=n_estimators,
        cuda=True,
    )

    # Set the Optimizer
    model.set_optimizer(
        "SGD", lr=lr, weight_decay=weight_decay, momentum=momentum
    )

    # Train and Evaluate
    model.fit(train_loader, epochs=epochs, test_loader=test_loader, tb_logger=tb_logger)

    if tb_logger:
        tb_logger.close()
