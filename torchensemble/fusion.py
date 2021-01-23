"""
  In fusion-based ensemble, predictions from all base estimators are
  first aggregated as an average output. After then, the training loss is
  computed based on this average output and the ground-truth. The training loss
  is then back-propagated to all base estimators simultaneously.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseModule, torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op


__all__ = ["FusionClassifier",
           "FusionRegressor"]


@torchensemble_model_doc("""Implementation on the FusionClassifier.""",
                         "model")
class FusionClassifier(BaseModule):

    def _forward(self, x):
        """
        Implementation on the internal data forwarding in FusionClassifier.
        """
        # Average
        outputs = [estimator(x) for estimator in self.estimators_]
        output = op.average(outputs)

        return output

    @torchensemble_model_doc(
        """Implementation on the data forwarding in FusionClassifier.""",
        "classifier_forward")
    def forward(self, x):
        output = self._forward(x)
        proba = F.softmax(output, dim=1)

        return proba

    @torchensemble_model_doc(
        """Set the attributes on optimizer for FusionClassifier.""",
        "set_optimizer")
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @torchensemble_model_doc(
        """Set the attributes on scheduler for FusionClassifier.""",
        "set_scheduler")
    def set_scheduler(self, scheduler_name, **kwargs):
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    @torchensemble_model_doc(
        """Implementation on the training stage of FusionClassifier.""",
        "fit")
    def fit(self,
            train_loader,
            epochs=100,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):

        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader, True)
        optimizer = set_module.set_optimizer(self,
                                             self.optimizer_name,
                                             **self.optimizer_args)

        # Set the scheduler if `set_scheduler` was called before
        if self.use_scheduler_:
            self.scheduler_ = set_module.set_scheduler(optimizer,
                                                       self.scheduler_name,
                                                       **self.scheduler_args)

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.

        # Training loop
        for epoch in range(epochs):
            self.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self._forward(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        _, predicted = torch.max(output.data, 1)
                        correct = (predicted == target).sum().item()

                        msg = ("Epoch: {:03d} | Batch: {:03d} | Loss:"
                               " {:.5f} | Correct: {:d}/{:d}")
                        self.logger.info(
                            msg.format(
                                epoch, batch_idx, loss, correct, data.size(0)
                                )
                            )

            # Validation
            if test_loader:
                self.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for _, (data, target) in enumerate(test_loader):
                        data = data.to(self.device)
                        target = target.to(self.device)
                        output = self.forward(data)
                        _, predicted = torch.max(output.data, 1)
                        correct += (predicted == target).sum().item()
                        total += target.size(0)
                    acc = 100 * correct / total

                    if acc > best_acc:
                        best_acc = acc
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = ("Epoch: {:03d} | Validation Acc: {:.3f}"
                           " % | Historical Best: {:.3f} %")
                    self.logger.info(msg.format(epoch, acc, best_acc))

            # Update the scheduler
            if hasattr(self, "scheduler_"):
                self.scheduler_.step()

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of FusionClassifier.""",
        "classifier_predict")
    def predict(self, test_loader):
        self.eval()
        correct = 0
        total = 0

        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        acc = 100 * correct / total

        return acc


@torchensemble_model_doc("""Implementation on the FusionRegressor.""",
                         "model")
class FusionRegressor(BaseModule):

    @torchensemble_model_doc(
        """Implementation on the data forwarding in FusionRegressor.""",
        "regressor_forward")
    def forward(self, x):
        outputs = [estimator(x) for estimator in self.estimators_]
        pred = op.average(outputs)

        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for FusionRegressor.""",
        "set_optimizer")
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @torchensemble_model_doc(
        """Set the attributes on scheduler for FusionRegressor.""",
        "set_scheduler")
    def set_scheduler(self, scheduler_name, **kwargs):
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    @torchensemble_model_doc(
        """Implementation on the training stage of FusionRegressor.""",
        "fit")
    def fit(self,
            train_loader,
            epochs=100,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):
        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader, False)
        optimizer = set_module.set_optimizer(self,
                                             self.optimizer_name,
                                             **self.optimizer_args)

        # Set the scheduler if `set_scheduler` was called before
        if self.use_scheduler_:
            self.scheduler_ = set_module.set_scheduler(optimizer,
                                                       self.scheduler_name,
                                                       **self.scheduler_args)

        # Utils
        criterion = nn.MSELoss()
        best_mse = float("inf")

        # Training loop
        for epoch in range(epochs):
            self.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        self.logger.info(msg.format(epoch, batch_idx, loss))

            # Validation
            if test_loader:
                self.eval()
                with torch.no_grad():
                    mse = 0
                    for _, (data, target) in enumerate(test_loader):
                        data = data.to(self.device)
                        target = target.to(self.device)
                        output = self.forward(data)
                        mse += criterion(output, target)
                    mse /= len(test_loader)

                    if mse < best_mse:
                        best_mse = mse
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = ("Epoch: {:03d} | Validation MSE: {:.5f} |"
                           " Historical Best: {:.5f}")
                    self.logger.info(msg.format(epoch, mse, best_mse))

            # Update the scheduler
            if hasattr(self, "scheduler_"):
                self.scheduler_.step()

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of FusionRegressor.""",
        "regressor_predict")
    def predict(self, test_loader):
        self.eval()
        mse = 0
        criterion = nn.MSELoss()

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            mse += criterion(output, target)

        return mse / len(test_loader)
