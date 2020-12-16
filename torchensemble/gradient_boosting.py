"""
  Gradient boosting is a classic sequential ensemble method. At each iteration,
  the learning target of a new base estimator is to fit the pseudo residual
  computed based on the ground truth and the output from base estimators
  fitted before, using ordinary least square.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseModule
from . import utils


class BaseGradientBoosting(BaseModule):

    def __init__(self,
                 estimator,
                 n_estimators,
                 estimator_args=None,
                 shrinkage_rate=1.,
                 cuda=True):
        """
        Parameters
        ----------
        estimator : torch.nn.Module
            The class of base estimator inherited from ``torch.nn.Module``.
        n_estimators : int
            The number of base estimators in the ensemble.
        estimator_args : dict, default=None
            The dictionary of parameters used to instantiate base estimators.
        shrinkage_rate : float, default=1
            The shrinkage rate of each base estimator in gradient boosting.
        cuda : bool, default=True
            - If ``True``, use GPU to train and evaluate the ensemble.
            - If ``False``, use CPU to train and evaluate the ensemble.

        Attributes
        ----------
        estimators_ : torch.nn.ModuleList
            An internal container that stores all base estimators.
        """
        super(BaseModule, self).__init__()

        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args
        self.shrinkage_rate = shrinkage_rate
        self.device = torch.device("cuda" if cuda else "cpu")

        self.estimators_ = nn.ModuleList()  # internal container

    def _validate_parameters(self, lr, weight_decay, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if not lr > 0:
            msg = ("The learning rate of optimizer = {} should be strictly"
                   " positive.")
            raise ValueError(msg.format(lr))

        if not weight_decay >= 0:
            msg = "The weight decay of optimizer = {} should not be negative."
            raise ValueError(msg.format(weight_decay))

        if not epochs > 0:
            msg = ("The number of training epochs = {} should be strictly"
                   " positive.")
            raise ValueError(msg.format(epochs))
        
        if not log_interval > 0:
            msg = ("The number of batches to wait before printting the"
                   " training status should be strictly positive, but got {}"
                   " instead.")
            raise ValueError(msg.format(log_interval))

        if not 0 < self.shrinkage_rate <= 1:
            msg = ('The shrinkage rate should be in the range (0, 1], but got'
                   ' {} instead.')
            raise ValueError(msg.format(self.shrinkage_rate))

    def fit(self,
            train_loader,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            optimizer="Adam",
            log_interval=100):
        """
        Implementation on the training stage of Gradient Boosting.

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
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, True)

        self.train()
        self._validate_parameters(lr, weight_decay, epochs, log_interval)
        criterion = nn.MSELoss(reduction="sum")

        # Base estimators are fitted sequentially in gradient boosting
        for est_idx, estimator in enumerate(self.estimators_):

            # Initialize an independent optimizer for each base estimator to
            # avoid unexpected dependencies.
            optimizer = utils.set_optimizer(estimator,
                                            optimizer,
                                            lr,
                                            weight_decay)

            # Training loop
            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(train_loader):

                    data, target = data.to(self.device), target.to(self.device)

                    # Learning target of the current estimator
                    residual = self._pseudo_residual(data, target, est_idx)

                    output = estimator(data)
                    loss = criterion(output, residual)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Print training status
                    if batch_idx % log_interval == 0:
                        msg = ('Estimator: {:03d} | Epoch: {:03d} | Batch:'
                               ' {:03d} | RegLoss: {:.5f}')
                        print(msg.format(est_idx, epoch, batch_idx, loss))


class GradientBoostingClassifier(BaseGradientBoosting):

    def _onehot_coding(self, target):
        """Convert the class label to a one-hot encoded vector."""
        target = target.view(-1)
        target_onehot = torch.FloatTensor(
            target.size()[0], self.n_outputs).to(self.device)
        target_onehot.data.zero_()
        target_onehot.scatter_(1, target.view(-1, 1), 1)

        return target_onehot

    # TODO: Store the output of *fitted* base estimators to avoid repeated data
    # forwarding. Since samples in the data loader can be shuffled, it requires
    # the index of each sample in the original dataset to be kept in memory.

    def _pseudo_residual(self, X, y, est_idx):
        """Compute pseudo residuals in classification."""
        y_onehot = self._onehot_coding(y)
        output = torch.zeros_like(y_onehot).to(self.device)

        # Before training the first estimator, we assume that GBM returns 0
        # for any input (i.e., null output).
        if est_idx == 0:
            return y_onehot - F.softmax(output, dim=1)
        else:
            for idx in range(est_idx):
                output += self.shrinkage_rate * self.estimators_[idx](X)

            return y_onehot - F.softmax(output, dim=1)

    def forward(self, X):
        """
        Implementation on the data forwarding in GradientBoostingClassifier.

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
        pred = torch.zeros(batch_size, self.n_outputs).to(self.device)

        # Take the average over class distributions from all base estimators.
        for estimator in self.estimators_:
            pred += self.shrinkage_rate * estimator(X)
        proba = F.softmax(pred, dim=1)

        return proba

    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of GradientBoostingClassifier.

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


class GradientBoostingRegressor(BaseGradientBoosting):

    def _pseudo_residual(self, X, y, est_idx):
        """Compute pseudo residuals in regression."""
        output = torch.zeros_like(y).to(self.device)

        # Before training the first estimator, we assume that GBM returns 0
        # for any input (i.e., null output).
        if est_idx == 0:
            return y
        else:
            for idx in range(est_idx):
                output += self.shrinkage_rate * self.estimators_[idx](X)

            return y - output

    def forward(self, X):
        """
        Implementation on the data forwarding in GradientBoostingRegressor.

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

        # Take the average over class distributions from all base estimators.
        for estimator in self.estimators_:
            pred += self.shrinkage_rate * estimator(X)

        return pred

    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of GradientBoostingRegressor.

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
