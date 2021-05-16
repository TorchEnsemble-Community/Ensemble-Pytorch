"""
  In soft gradient boosting, all base estimators could be simulataneously
  fitted, while achieveing the similar boosting improvements as in gradient
  boosting.
"""


import abc
import torch
import logging
import warnings
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from ._base import BaseModule, BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op
from .utils.logging import get_tb_logger


# __all__ = ["SoftGradientBoostingClassifier", "SoftGradientBoostingRegressor"]


def _parallel_compute_pseudo_residual(
    output, target, estimator_idx, shrinkage_rate, n_outputs, is_classification
):
    """
    Compute pseudo residuals defined in sGBM for each base estimator in a
    parallel fashion.
    """
    accumulated_output = torch.zeros_like(output, device=output.device)
    for i in range(estimator_idx):
        accumulated_output += shrinkage_rate * output[i]

    # Classification
    if is_classification:
        residual = op.pseudo_residual_classification(
            target, accumulated_output, n_outputs
        )
    # Regression
    else:
        residual = op.pseudo_residual_regression(target, accumulated_output)

    return residual


class _BaseSoftGradientBoosting(BaseModule):
    def __init__(
        self,
        estimator,
        n_estimators,
        estimator_args=None,
        shrinkage_rate=1.0,
        cuda=True,
        n_jobs=None,
    ):
        super(BaseModule, self).__init__()
        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args

        if estimator_args and not isinstance(estimator, type):
            msg = (
                "The input `estimator_args` will have no effect since"
                " `estimator` is already an object after instantiation."
            )
            warnings.warn(msg, RuntimeWarning)

        self.shrinkage_rate = shrinkage_rate
        self.device = torch.device("cuda" if cuda else "cpu")
        self.n_jobs = n_jobs
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()

        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def _validate_parameters(self, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if not epochs > 0:
            msg = (
                "The number of training epochs = {} should be strictly"
                " positive."
            )
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))

        if not log_interval > 0:
            msg = (
                "The number of batches to wait before printting the"
                " training status should be strictly positive, but got {}"
                " instead."
            )
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))

        if not 0 < self.shrinkage_rate <= 1:
            msg = (
                "The shrinkage rate should be in the range (0, 1], but got"
                " {} instead."
            )
            self.logger.error(msg.format(self.shrinkage_rate))
            raise ValueError(msg.format(self.shrinkage_rate))

    @abc.abstractmethod
    def _evaluate_during_fit(self, test_loader, epoch):
        """Evaluate the ensemble after each training epoch."""

    def fit(
        self,
        train_loader,
        epochs=100,
        use_reduction_sum=True,
        log_interval=100,
        test_loader=None,
        early_stopping_rounds=2,
        save_model=True,
        save_dir=None,
    ):

        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Utils
        criterion = (
            nn.MSELoss(reduction="sum") if use_reduction_sum else nn.MSELoss()
        )
        total_iters = 0

        # Set up optimizer and learning rate scheduler
        optimizer = set_module.set_optimizer(
            self, self.optimizer_name, **self.optimizer_args
        )

        if self.use_scheduler_:
            scheduler = set_module.set_scheduler(
                optimizer,
                self.scheduler_name,
                **self.scheduler_args  # noqa: E501
            )

        for epoch in range(epochs):
            self.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(self.device), target.to(self.device)
                output = [estimator(data) for estimator in self.estimators_]

                # Compute pseudo residuals in parallel
                rets = Parallel(n_jobs=self.n_jobs)(
                    delayed(_parallel_compute_pseudo_residual)(
                        output,
                        target,
                        i,
                        self.shrinkage_rate,
                        self.n_outputs,
                        self.is_classification,
                    )
                    for i in range(self.n_estimators)
                )

                # Compute sGBM loss
                loss = torch.tensor(0.0, device=self.device)
                for idx, estimator in enumerate(self.estimators_):
                    loss += criterion(output[idx], rets[idx])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        self.logger.info(msg.format(epoch, batch_idx, loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "sGBM/Train_Loss", loss, total_iters
                            )
                total_iters += 1

            # Validation
            if test_loader:
                self._evaluate_during_fit(test_loader, epoch)

            # Update the scheduler
            if self.use_scheduler_:
                scheduler.step()

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)


class SoftGradientBoostingClassifier(
    _BaseSoftGradientBoosting, BaseClassifier
):
    def __init__(
        self,
        estimator,
        n_estimators,
        estimator_args=None,
        shrinkage_rate=1.0,
        cuda=True,
        n_jobs=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_args=estimator_args,
            shrinkage_rate=shrinkage_rate,
            cuda=cuda,
            n_jobs=n_jobs,
        )
        self.is_classification = True


class SoftGradientBoostingRegressor(_BaseSoftGradientBoosting, BaseRegressor):
    def __init__(
        self,
        estimator,
        n_estimators,
        estimator_args=None,
        shrinkage_rate=1.0,
        cuda=True,
        n_jobs=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_args=estimator_args,
            shrinkage_rate=shrinkage_rate,
            cuda=cuda,
            n_jobs=n_jobs,
        )
        self.is_classification = False
