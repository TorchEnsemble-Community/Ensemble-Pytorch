"""
  In voting-based ensemble methods, each base estimator is trained
  independently, and the final prediction takes the average over predictions
  from all base estimators.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from ._base import BaseModule


# TODO: Memory optimization by registering read-only objects into shared memory
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

        output = estimator(X_train)
        loss = criterion(output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            if is_classification:
                y_pred = output.data.max(1)[1]
                correct = y_pred.eq(y_train.view(-1).data).sum()

                msg = ('Estimator: {:03d} | Epoch: {:03d} |'
                       ' Batch: {:03d} | Loss: {:.5f} | Correct:'
                       ' {:d}/{:d}')
                print(msg.format(estimator_idx, epoch, batch_idx, loss,
                                 correct, batch_size))
            else:
                msg = ('Estimator: {:03d} | Epoch: {:03d} |'
                       ' Batch: {:03d} | Loss: {:.5f}')
                print(msg.format(estimator_idx, epoch, batch_idx, loss))

    return estimator


class VotingClassifier(BaseModule):

    def forward(self, X):
        batch_size = X.size()[0]
        y_pred_proba = torch.zeros(batch_size, self.output_dim).to(self.device)

        # Average over class probabilities from all base estimators.
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


class VotingRegressor(BaseModule):

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
