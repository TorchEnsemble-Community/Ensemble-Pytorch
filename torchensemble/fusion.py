"""
  In fusion-based ensemble methods, predictions from all base estimators are
  first aggregated as an average output. After then, the training loss is
  computed based on this average output and the ground-truth. The training loss
  is then back-propagated to all base estimators simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseModule, torchensemble_model_doc
from . import utils


__author__ = ["Yi-Xuan Xu"]
__all__ = ["FusionClassifier",
           "FusionRegressor"]


@torchensemble_model_doc("""Implementation on the FusionClassifier.""",
                         "model")
class FusionClassifier(BaseModule):

    def _forward(self, X):
        """
        Implementation on the internal data forwarding in FusionClassifier.
        """
        batch_size = X.size()[0]
        proba = torch.zeros(batch_size, self.n_outputs).to(self.device)

        # Average
        for estimator in self.estimators_:
            proba += estimator(X) / self.n_estimators

        return proba

    @torchensemble_model_doc(
        """Implementation on the data forwarding in FusionClassifier.""",
        "classifier_forward")
    def forward(self, X):
        proba = self._forward(X)

        return F.softmax(proba, dim=1)

    @torchensemble_model_doc(
        """Implementation on the training stage of FusionClassifier.""",
        "fit")
    def fit(self,
            train_loader,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            optimizer="Adam",
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):

        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, True)
        self._validate_parameters(lr, weight_decay, epochs, log_interval)
        optimizer = utils.set_optimizer(self, optimizer, lr, weight_decay)

        self.train()

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.

        # Training loop
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):

                batch_size = data.size()[0]
                data, target = data.to(self.device), target.to(self.device)

                output = self._forward(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        pred = output.data.max(1)[1]
                        correct = pred.eq(target.view(-1).data).sum()

                        msg = ("Epoch: {:03d} | Batch: {:03d} | Loss:"
                               " {:.5f} | Correct: {:d}/{:d}")
                        self.logger.info(
                            msg.format(
                                epoch, batch_idx, loss, correct, batch_size
                                )
                            )

            # Validation
            if test_loader:
                with torch.no_grad():
                    correct = 0.
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = (data.to(self.device),
                                        target.to(self.device))
                        output = self.forward(data)
                        pred = output.data.max(1)[1]
                        correct += pred.eq(target.view(-1).data).sum()
                    acc = 100. * float(correct) / len(test_loader.dataset)

                    if acc > best_acc:
                        best_acc = acc
                        if save_model:
                            utils.save(self, save_dir, self.logger)

                    msg = ("Epoch: {:03d} | Validation Acc: {:.3f}"
                           " % | Historical Best: {:.3f} %")
                    self.logger.info(msg.format(epoch, acc, best_acc))

        if save_model and not test_loader:
            utils.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of FusionClassifier.""",
        "classifier_predict")
    def predict(self, test_loader):
        self.eval()
        correct = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()

        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy


@torchensemble_model_doc("""Implementation on the FusionRegressor.""",
                         "model")
class FusionRegressor(BaseModule):

    @torchensemble_model_doc(
        """Implementation on the data forwarding in FusionRegressor.""",
        "regressor_forward")
    def forward(self, X):
        batch_size = X.size()[0]
        pred = torch.zeros(batch_size, self.n_outputs).to(self.device)

        for estimator in self.estimators_:
            pred += estimator(X) / self.n_estimators

        return pred

    @torchensemble_model_doc(
        """Implementation on the training stage of FusionRegressor.""",
        "fit")
    def fit(self,
            train_loader,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            optimizer="Adam",
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):
        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, False)
        self._validate_parameters(lr, weight_decay, epochs, log_interval)
        optimizer = utils.set_optimizer(self, optimizer, lr, weight_decay)

        self.train()

        # Utils
        criterion = nn.MSELoss()
        best_mse = float("inf")

        # Training loop
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(self.device), target.to(self.device)

                output = self.forward(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        self.logger.info(msg.format(epoch, batch_idx, loss))

            # Validation
            if test_loader:
                with torch.no_grad():
                    mse = 0.
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = (data.to(self.device),
                                        target.to(self.device))
                        output = self.forward(data)
                        mse += criterion(output, target)
                    mse /= len(test_loader)

                    if mse < best_mse:
                        best_mse = mse
                        if save_model:
                            utils.save(self, save_dir, self.logger)

                    msg = ("Epoch: {:03d} | Validation MSE: {:.5f} |"
                           " Historical Best: {:.5f}")
                    self.logger.info(msg.format(epoch, mse, best_mse))

        if save_model and not test_loader:
            utils.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of FusionRegressor.""",
        "regressor_predict")
    def predict(self, test_loader):
        self.eval()
        mse = 0.
        criterion = nn.MSELoss()

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)

            mse += criterion(output, target)

        return mse / len(test_loader)
