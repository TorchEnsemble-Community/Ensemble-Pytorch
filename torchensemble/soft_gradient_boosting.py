"""
  In soft gradient boosting, all base estimators could be simultaneously
  fitted, while achieving the similar boosting improvements as in gradient
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


__all__ = ["SoftGradientBoostingClassifier", "SoftGradientBoostingRegressor"]


__model_doc = """
    Parameters
    ----------
    estimator : torch.nn.Module
        The class or object of your base estimator.

        - If :obj:`class`, it should inherit from :mod:`torch.nn.Module`.
        - If :obj:`object`, it should be instantiated from a class inherited
          from :mod:`torch.nn.Module`.
    n_estimators : int
        The number of base estimators in the ensemble.
    estimator_args : dict, default=None
        The dictionary of hyper-parameters used to instantiate base
        estimators. This parameter will have no effect if ``estimator`` is a
        base estimator object after instantiation.
    shrinkage_rate : float, default=1
        The shrinkage rate used in gradient boosting.
    cuda : bool, default=True

        - If ``True``, use GPU to train and evaluate the ensemble.
        - If ``False``, use CPU to train and evaluate the ensemble.
    n_jobs : int, default=None
        The number of workers for training the ensemble. This input
        argument is used for parallel ensemble methods such as
        :mod:`voting` and :mod:`bagging`. Setting it to an integer larger
        than ``1`` enables ``n_jobs`` base estimators to be trained
        simultaneously.

    Attributes
    ----------
    estimators_ : torch.nn.ModuleList
        An internal container that stores all fitted base estimators.
"""


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A data loader that contains the training data.
    epochs : int, default=100
        The number of training epochs per base estimator.
    use_reduction_sum : bool, default=True
        Whether to set ``reduction="sum"`` for the internal mean squared
        error used to fit each base estimator.
    log_interval : int, default=100
        The number of batches to wait before logging the training status.
    test_loader : torch.utils.data.DataLoader, default=None
        A data loader that contains the evaluating data.

        - If ``None``, no validation is conducted after each base
          estimator being trained.
        - If not ``None``, the ensemble will be evaluated on this
          dataloader after each base estimator being trained.
    save_model : bool, default=True
        Specify whether to save the model parameters.

        - If test_loader is ``None``, the ensemble containing
          ``n_estimators`` base estimators will be saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model parameters.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
"""


def _soft_gradient_boosting_model_doc(header, item="model"):
    """
    Decorator on obtaining documentation for different gradient boosting
    models.
    """

    def get_doc(item):
        """Return the selected item"""
        __doc = {"model": __model_doc, "fit": __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + "\n\n"]
        doc.extend(get_doc(item))
        cls.__doc__ = "".join(doc)
        return cls

    return adddoc


def _parallel_compute_pseudo_residual(
    output, target, estimator_idx, shrinkage_rate, n_outputs, is_classification
):
    """
    Compute pseudo residuals in soft gradient boosting for each base estimator
    in a parallel fashion.
    """
    accumulated_output = torch.zeros_like(output[0], device=output[0].device)
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
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)
                output = [estimator(*data) for estimator in self.estimators_]

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | RegLoss: {:.5f}"
                        self.logger.info(msg.format(epoch, batch_idx, loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "sGBM/Train_Loss", loss, total_iters
                            )
                total_iters += 1

            # Validation
            if test_loader:
                flag = self._evaluate_during_fit(test_loader, epoch)
                if save_model and flag:
                    io.save(self, save_dir, self.logger)

            # Update the scheduler
            if self.use_scheduler_:
                scheduler.step()

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)


@_soft_gradient_boosting_model_doc(
    """Implementation on the SoftGradientBoostingClassifier.""", "model"
)
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
        self._criterion = nn.CrossEntropyLoss()
        self.is_classification = True
        self.best_acc = 0.0

    @torch.no_grad()
    def _evaluate_during_fit(self, test_loader, epoch):
        self.eval()
        correct = 0
        total = 0
        flag = False
        for _, elem in enumerate(test_loader):
            data, target = io.split_data_target(elem, self.device)
            output = self.forward(*data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
        acc = 100 * correct / total

        if acc > self.best_acc:
            self.best_acc = acc
            flag = True

        msg = (
            "Epoch: {:03d} | Validation Acc: {:.3f}"
            " % | Historical Best: {:.3f} %"
        )
        self.logger.info(msg.format(epoch, acc, self.best_acc))
        if self.tb_logger:
            self.tb_logger.add_scalar(
                "soft_gradient_boosting/Validation_Acc", acc, epoch
            )

        return flag

    @torchensemble_model_doc(
        """Set the attributes on optimizer for SoftGradientBoostingClassifier.""",  # noqa: E501
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for SoftGradientBoostingClassifier.""",  # noqa: E501
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for SoftGradientBoostingClassifier.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @_soft_gradient_boosting_model_doc(
        """Implementation on the training stage of SoftGradientBoostingClassifier.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        use_reduction_sum=True,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        super().fit(
            train_loader=train_loader,
            epochs=epochs,
            use_reduction_sum=use_reduction_sum,
            log_interval=log_interval,
            test_loader=test_loader,
            save_model=save_model,
            save_dir=save_dir,
        )

    @torchensemble_model_doc(
        """Implementation on the data forwarding in SoftGradientBoostingClassifier.""",  # noqa: E501
        "classifier_forward",
    )
    def forward(self, *x):
        output = [estimator(*x) for estimator in self.estimators_]
        output = op.sum_with_multiplicative(output, self.shrinkage_rate)
        proba = F.softmax(output, dim=1)

        return proba

    @torchensemble_model_doc(item="classifier_evaluate")
    def evaluate(self, test_loader, return_loss=False):
        return super().evaluate(test_loader, return_loss)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)


@_soft_gradient_boosting_model_doc(
    """Implementation on the SoftGradientBoostingRegressor.""", "model"
)
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
        self._criterion = nn.MSELoss()
        self.is_classification = False
        self.best_mse = float("inf")

    @torch.no_grad()
    def _evaluate_during_fit(self, test_loader, epoch):
        self.eval()
        mse = 0.0
        flag = False
        criterion = nn.MSELoss()
        for _, elem in enumerate(test_loader):
            data, target = io.split_data_target(elem, self.device)
            output = self.forward(*data)
            mse += criterion(output, target)
        mse /= len(test_loader)

        if mse < self.best_mse:
            self.best_mse = mse
            flag = True

        msg = (
            "Epoch: {:03d} | Validation MSE: {:.5f} | Historical Best: {:.5f}"
        )
        self.logger.info(msg.format(epoch, mse, self.best_mse))
        if self.tb_logger:
            self.tb_logger.add_scalar(
                "soft_gradient_boosting/Validation_MSE", mse, epoch
            )

        return flag

    @torchensemble_model_doc(
        """Set the attributes on optimizer for SoftGradientBoostingRegressor.""",  # noqa: E501
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for SoftGradientBoostingRegressor.""",  # noqa: E501
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for SoftGradientBoostingRegressor.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @_soft_gradient_boosting_model_doc(
        """Implementation on the training stage of SoftGradientBoostingRegressor.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        use_reduction_sum=True,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        super().fit(
            train_loader=train_loader,
            epochs=epochs,
            use_reduction_sum=use_reduction_sum,
            log_interval=log_interval,
            test_loader=test_loader,
            save_model=save_model,
            save_dir=save_dir,
        )

    @torchensemble_model_doc(
        """Implementation on the data forwarding in SoftGradientBoostingRegressor.""",  # noqa: E501
        "regressor_forward",
    )
    def forward(self, *x):
        outputs = [estimator(*x) for estimator in self.estimators_]
        pred = op.sum_with_multiplicative(outputs, self.shrinkage_rate)

        return pred

    @torchensemble_model_doc(item="regressor_evaluate")
    def evaluate(self, test_loader):
        return super().evaluate(test_loader)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)
