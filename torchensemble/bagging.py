"""
  In bagging-based ensemble methods, each base estimator is trained
  independently. In addition, sampling with replacement is conducted on the
  training data to further encourge the diversity between different base
  estimators in the ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import (Parallel, delayed)

from ._base import BaseModule
from . import utils


def _parallel_fit(
        train_loader,
        lr,
        weight_decay,
        epochs,
        optimizer,
        log_interval,
        idx,
        estimator,
        criterion,
        device,
        is_classification=True
):
    """Private function used to fit base estimators in parallel."""
    optimizer = utils.set_optimizer(estimator, optimizer, lr, weight_decay)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):

            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)

            # In `BaggingClassifier`, each base estimator is fitted on a batch
            # of data after sampling with replacement.
            sampling_mask = torch.randint(high=batch_size,
                                          size=(int(batch_size),),
                                          dtype=torch.int64)
            sampling_mask = torch.unique(sampling_mask)  # remove duplicates
            sampling_data = data[sampling_mask]
            sampling_target = target[sampling_mask]

            sampling_output = estimator(sampling_data)
            loss = criterion(sampling_output, sampling_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training status
            if batch_idx % log_interval == 0:

                if is_classification:
                    pred = sampling_output.data.max(1)[1]
                    correct = pred.eq(sampling_target.view(-1).data).sum()

                    msg = ("Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                           " | Loss: {:.5f} | Correct: {:d}/{:d}")
                    print(msg.format(idx, epoch, batch_idx, loss,
                                     correct, sampling_data.size()[0]))
                else:
                    msg = ("Estimator: {:03d} | Epoch: {:03d} |Batch: {:03d}"
                           " | Loss: {:.5f}")
                    print(msg.format(idx, epoch, batch_idx, loss))

    return estimator


class BaggingClassifier(BaseModule):

    def forward(self, X):
        """
        Implementation on the data forwarding in BaggingClassifier.

        Parameters
        ----------
        X : tensor
            Input batch of data, which should be a valid input data batch for
            base estimators.

        Returns
        -------
        proba : tensor of shape (batch_size, n_classes)
            The predicted class distribution.
        """
        batch_size = X.size()[0]
        proba = torch.zeros(batch_size, self.n_outputs).to(self.device)

        # Take the average over class distributions from all base estimators.
        for estimator in self.estimators_:
            proba += F.softmax(estimator(X), dim=1) / self.n_estimators

        return proba

    def fit(self,
            train_loader,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            optimizer="Adam",
            log_interval=100):
        """
        Implementation on the training stage of BaggingClassifier.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the training data.
        lr : float, default=1e-3
            The learning rate of the parameter optimizer.
        weight_decay : float, default=5e-4
            The weight decay of the parameter optimizer.
        epochs : int, default=100
            The number of training epochs.
        optimizer : {"SGD", "Adam", "RMSprop"}, default="Adam"
            The type of parameter optimizer.
        log_interval : int, default=100
            The number of batches to wait before printting the training status.
        """

        # Instantiate base estimators and set attributes
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, True)

        self.train()
        self._validate_parameters(lr, weight_decay, epochs, log_interval)
        criterion = nn.CrossEntropyLoss()

        rets = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit)(
                train_loader,
                lr,
                weight_decay,
                epochs,
                optimizer,
                log_interval,
                idx,
                estimator,
                criterion,
                self.device,
                True
            )
            for idx, estimator in enumerate(estimators)
        )

        self.estimators_.extend(rets)

    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of BaggingClassifier.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the testing data.

        Returns
        -------
        accuracy : float
            The testing accuracy of the fitted model on the ``test_loader``.
        """
        self.eval()
        correct = 0.

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()

        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy


class BaggingRegressor(BaseModule):

    def forward(self, X):
        """
        Implementation on the data forwarding in BaggingRegressor.

        Parameters
        ----------
        X : tensor
            Input batch of data, which should be a valid input data batch for
            base estimators.

        Returns
        -------
        pred : tensor of shape (batch_size, n_outputs)
            The predicted values.
        """
        batch_size = X.size()[0]
        pred = torch.zeros(batch_size, self.n_outputs).to(self.device)

        # Take the average over predictions from all base estimators.
        for estimator in self.estimators_:
            pred += estimator(X) / self.n_estimators

        return pred

    def fit(self,
            train_loader,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            optimizer="Adam",
            log_interval=100):
        """
        Implementation on the training stage of BaggingRegressor.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the training data.
        lr : float, default=1e-3
            The learning rate of the parameter optimizer.
        weight_decay : float, default=5e-4
            The weight decay of the parameter optimizer.
        epochs : int, default=100
            The number of training epochs.
        optimizer : {"SGD", "Adam", "RMSprop"}, default="Adam"
            The type of parameter optimizer.
        log_interval : int, default=100
            The number of batches to wait before printting the training status.
        """

        # Instantiate base estimators and set attributes
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, False)

        self.train()
        self._validate_parameters(lr, weight_decay, epochs, log_interval)
        criterion = nn.MSELoss()

        rets = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit)(
                train_loader,
                lr,
                weight_decay,
                epochs,
                optimizer,
                log_interval,
                idx,
                estimator,
                criterion,
                self.device,
                False
            )
            for idx, estimator in enumerate(estimators)
        )

        self.estimators_.extend(rets)

    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of BaggingRegressor.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the testing data.

        Returns
        -------
        mse : float
            The testing mean squared error of the fitted model on the
            ``test_loader``.
        """
        self.eval()
        mse = 0.
        criterion = nn.MSELoss()

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)

            mse += criterion(output, target)

        return mse / len(test_loader)
