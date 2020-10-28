import sys
sys.path.append('../')

import time
import torch
from sklearn.preprocessing import scale
from sklearn.datasets import load_svmlight_file
from torch.utils.data import TensorDataset, DataLoader

from model.mlp import MLP
from ensemble.fusion import FusionRegressor
from ensemble.voting import VotingRegressor
from ensemble.bagging import BaggingRegressor
from ensemble.gradient_boosting import GradientBoostingRegressor


def load_data(batch_size):
    
    # The dataset can be downloaded from: 
    #   https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#YearPredictionMSD
    
    train = load_svmlight_file('../../Dataset/LIBSVM/YearPredictionMSD.bz2')
    test = load_svmlight_file('../../Dataset/LIBSVM/YearPredictionMSD.t.bz2')

    X_train, X_test = (torch.FloatTensor(train[0].toarray()), 
                       torch.FloatTensor(test[0].toarray()))
    y_train, y_test = (torch.FloatTensor(scale(train[1]).reshape(-1, 1)), 
                       torch.FloatTensor(scale(test[1]).reshape(-1, 1)))
    
    # Tensor -> Data loader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def display_records(records):
    msg = ('{:<28} | Testing MSE: {:.2f} | Training Time: {:.2f} s |'
           ' Evaluating Time: {:.2f} s')
    
    print('\n')
    for method, training_time, evaluating_time, mse in records:
        print(msg.format(method, mse, training_time, evaluating_time))


if __name__ == '__main__':
    
    # Hyper-parameters
    n_estimators = 10
    output_dim = 1
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 50
    
    # Utils
    batch_size = 256
    records = []
    torch.manual_seed(0)
    
    # Load data
    train_loader, test_loader = load_data(batch_size)
    print('Finish loading data...\n')

    # FusionRegressor
    model = FusionRegressor(estimator=MLP,
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
    testing_mse = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(('FusionRegressor', 
                    training_time, 
                    evaluating_time, 
                    testing_mse))
    
    # VotingRegressor
    model = VotingRegressor(estimator=MLP,
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
    testing_mse = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(('VotingRegressor', 
                    training_time, 
                    evaluating_time, 
                    testing_mse))

    # BaggingRegressor
    model = BaggingRegressor(estimator=MLP,
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
    testing_mse = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(('BaggingRegressor', 
                    training_time, 
                    evaluating_time, 
                    testing_mse))
    
    # GradientBoostingRegressor
    model = GradientBoostingRegressor(estimator=MLP,
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
    testing_mse = model.predict(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    records.append(('GradientBoostingRegressor', 
                    training_time, 
                    evaluating_time, 
                    testing_mse))

    display_records(records)
