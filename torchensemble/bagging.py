"""
  In bagging-based ensemble methods, each base estimator is trained
  independently. In addition, sampling with replacement is conducted on the
  training data to further encourge the diversity between different base
  estimators in the ensemble model.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from ._base import BaseModule


def _parallel_fit(epoch, estimator_idx,
                  estimator, data_loader, criterion, lr, weight_decay,
                  device, log_interval, is_classification=True):
    """
    Private function used to fit base estimators in parallel.
    """
    optimizer = torch.optim.Adam(estimator.parameters(),
                                 lr=lr, weight_decay=weight_decay)

    for batch_idx, (X_train, y_train) in enumerate(data_loader):

        batch_size = X_train.size()[0]
        X_train, y_train = (X_train.to(device),
                            y_train.to(device))

        # In `BaggingClassifier`, each base estimator is fitted on a batch of
        # data after sampling with replacement.
        sampling_mask = torch.randint(high=batch_size,
                                      size=(int(batch_size),),
                                      dtype=torch.int64)
        sampling_mask = torch.unique(sampling_mask)  # remove duplicates
        sampling_X_train = X_train[sampling_mask]
        sampling_y_train = y_train[sampling_mask]

        sampling_output = estimator(sampling_X_train)
        loss = criterion(sampling_output, sampling_y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            if is_classification:
                y_pred = sampling_output.data.max(1)[1]
                correct = y_pred.eq(sampling_y_train.view(-1).data).sum()

                msg = ('Estimator: {:03d} | Epoch: {:03d} |'
                       ' Batch: {:03d} | Loss: {:.5f} | Correct:'
                       ' {:03d}/{:03d}')
                print(msg.format(estimator_idx, epoch, batch_idx, loss,
                                 correct, sampling_X_train.size()[0]))
            else:
                msg = ('Estimator: {:03d} | Epoch: {:03d} |'
                       ' Batch: {:03d} | Loss: {:.5f}')
                print(msg.format(estimator_idx, epoch, batch_idx, loss))

    return estimator


class BaggingClassifier(BaseModule):

    def forward(self, X):
        batch_size = X.size()[0]
        y_pred_proba = torch.zeros(batch_size, self.output_dim).to(self.device)

        # Average over class distributions predicted from all base estimators
        for estimator in self.estimators_:
            y_pred_proba += F.softmax(estimator(X), dim=1)
        y_pred_proba /= self.n_estimators

        return y_pred_proba

    def fit(self, train_loader):

        self.train()
        self._validate_parameters()
        criterion = nn.CrossEntropyLoss()

        # Create a pool of workers for repeated calls to the joblib.Parallel
        with Parallel(n_jobs=self.n_jobs) as parallel:

            for epoch in range(self.epochs):

                rets = parallel(delayed(_parallel_fit)(
                    epoch, idx, estimator, train_loader, criterion,
                    self.lr, self.weight_decay, self.device, self.log_interval)
                    for idx, estimator in enumerate(self.estimators_))

                # Update the base estimator container
                for i in range(self.n_estimators):
                    self.estimators_[i] = copy.deepcopy(rets[i])

    def predict(self, test_loader):

        self.eval()
        correct = 0.

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.forward(X_test)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(y_test.view(-1).data).sum()

        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy


class BaggingRegressor(BaseModule):

    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)

        # Average over predictions from all base estimators
        for estimator in self.estimators_:
            y_pred += estimator(X)
        y_pred /= self.n_estimators

        return y_pred

    def fit(self, train_loader):

        self.train()
        self._validate_parameters()
        criterion = nn.MSELoss()

        # Create a pool of workers for repeated calls to the joblib.Parallel
        with Parallel(n_jobs=self.n_jobs) as parallel:

            for epoch in range(self.epochs):

                rets = parallel(delayed(_parallel_fit)(
                    epoch, idx, estimator, train_loader, criterion,
                    self.lr, self.weight_decay, self.device,
                    self.log_interval, False)
                    for idx, estimator in enumerate(self.estimators_))

                # Update the base estimator container
                for i in range(self.n_estimators):
                    self.estimators_[i] = copy.deepcopy(rets[i])

    def predict(self, test_loader):

        self.eval()
        mse = 0.
        criterion = nn.MSELoss()

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.forward(X_test)

            mse += criterion(output, y_test)

        return mse / len(test_loader)
