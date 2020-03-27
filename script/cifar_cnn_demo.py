import sys
sys.path.append('../')
import torch
from model.lenet5 import LeNet5
from ensemble.votingclassifier import VotingClassifier
from ensemble.baggingclassifier import BaggingClassifier
from ensemble.gradientboostingclassifier import GradientBoostingClassifier
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    batch_size = 128
    data_dir = "../../Dataset/cifar"
    
    train_loader = DataLoader(datasets.CIFAR10(data_dir, train=True, download=True,
                                               transform=transforms.Compose([
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(32, 4),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(datasets.CIFAR10(data_dir, train=False,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                             batch_size=batch_size,
                             shuffle=True)
    
    ensemble_args = {"output_dim": 10,
                     "n_estimators": 10,
                     "cuda": True,
                     "epochs": 100,
                     "log_interval": 100,
                     "lr": 1e-3,
                     "weight_decay": 5e-4,
                     "shrinkage_rate": 0.3}
    
    learner_args = {}
    
    model = GradientBoostingClassifier(ensemble_args, LeNet5, learner_args)
    
    summary(model, (3, 32, 32))
    
    # model.fit(train_loader)
    # model.evaluate(test_loader)
