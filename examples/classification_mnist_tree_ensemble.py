import time
import torch
from torchvision import datasets, transforms

from torchensemble import NeuralForestClassifier
from torchensemble.utils.logging import set_logger


if __name__ == "__main__":

    # Hyper-parameters
    n_estimators = 5
    depth = 5
    lamda = 1e-3
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 50

    # Utils
    cuda = False
    n_jobs = 1
    batch_size = 128
    data_dir = "../../Dataset/mnist"  # MODIFY THIS IF YOU WANT

    # Load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    logger = set_logger(
        "classification_mnist_tree_ensemble", use_tb_logger=False
    )

    model = NeuralForestClassifier(
        n_estimators=n_estimators,
        depth=depth,
        lamda=lamda,
        cuda=cuda,
        n_jobs=-1,
    )

    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs, test_loader=test_loader)
    toc = time.time()
    training_time = toc - tic
