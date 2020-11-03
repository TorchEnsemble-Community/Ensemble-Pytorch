""" Example on classification using CIFAR-10. """

import sys
sys.path.append('../')

import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenet5 import LeNet5
from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier


def display_records(records):
    msg = ('{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |'
           ' Evaluating Time: {:.2f} s')
    
    print('\n')
    for method, training_time, evaluating_time, acc in records:
        print(msg.format(method, acc, training_time, evaluating_time))


if __name__ == '__main__':
    
    # Hyper-parameters
    n_estimators = 10
    output_dim = 10
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 100
    
    # Utils
    batch_size = 128
    data_dir = '../../Dataset/cifar'
    records = []
    torch.manual_seed(0)
    
    # Load data
    transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                           (0.2023, 0.1994, 0.2010))])
    
    train_loader = DataLoader(datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transformer), 
        batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(datasets.CIFAR10(
        data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))])),
        batch_size=batch_size, shuffle=True)
    
    # FusionClassifier
    model = FusionClassifier(estimator=LeNet5,
                              n_estimators=n_estimators,
                              output_dim=output_dim,
                              lr=lr,
                              weight_decay=weight_decay,
                              epochs=epochs)
    
    tic = time.time()
    model.fit(train_loader)
    toc = time.time()
    training_time = toc - tic
    
    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(('FusionClassifier', 
                    training_time, 
                    evaluating_time, 
                    testing_acc))
    
    # VotingClassifier
    model = VotingClassifier(estimator=LeNet5,
                             n_estimators=n_estimators,
                             output_dim=output_dim,
                             lr=lr,
                             weight_decay=weight_decay,
                             epochs=epochs,
                             n_jobs=1)
    
    tic = time.time()
    model.fit(train_loader)
    toc = time.time()
    training_time = toc - tic
    
    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(('VotingClassifier', 
                    training_time, 
                    evaluating_time, 
                    testing_acc))

    # BaggingClassifier
    model = BaggingClassifier(estimator=LeNet5,
                              n_estimators=n_estimators,
                              output_dim=output_dim,
                              lr=lr,
                              weight_decay=weight_decay,
                              epochs=epochs)
    
    tic = time.time()
    model.fit(train_loader)
    toc = time.time()
    training_time = toc - tic
    
    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(('BaggingClassifier', 
                    training_time, 
                    evaluating_time, 
                    testing_acc))
    
    # GradientBoostingClassifier
    model = GradientBoostingClassifier(estimator=LeNet5,
                                       n_estimators=n_estimators,
                                       output_dim=output_dim,
                                       lr=lr,
                                       weight_decay=weight_decay,
                                       epochs=epochs)
    
    tic = time.time()
    model.fit(train_loader)
    toc = time.time()
    training_time = toc - tic
    
    tic = time.time()
    testing_acc = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(('GradientBoostingClassifier', 
                    training_time, 
                    evaluating_time, 
                    testing_acc))

    # Print results on different ensemble methods
    display_records(records)
