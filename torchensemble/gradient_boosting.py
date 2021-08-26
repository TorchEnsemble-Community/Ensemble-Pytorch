"""
  Gradient boosting is a classic sequential ensemble method. At each iteration,
  the learning target of a new base estimator is to fit the pseudo residuals
  computed based on the ground truth and the output from base estimators
  fitted before, using ordinary least square.
"""


import abc
import torch
import logging
import warnings
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseModule, BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op
from .utils.logging import get_tb_logger


__all__ = ["GradientBoostingClassifier", "GradientBoostingRegressor"]


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
    early_stopping_rounds : int, default=2
        Specify the number of tolerant rounds for early stopping. When the
        validation performance of the ensemble does not improve after
        adding the base estimator fitted in current iteration, the internal
        counter on early stopping will increase by one. When the value of
        the internal counter reaches ``early_stopping_rounds``, the
        training stage will terminate instantly.
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


def _gradient_boosting_model_doc(header, item="model"):
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


class _BaseGradientBoosting(BaseModule):
    def __init__(
        self,
        estimator,
        n_estimators,
        estimator_args=None,
        shrinkage_rate=1.0,
        cuda=True,
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
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()

        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def _validate_parameters(
        self, epochs, log_interval, early_stopping_rounds
    ):
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

        if not early_stopping_rounds >= 1:
            msg = (
                "The number of tolerant rounds before triggering the"
                " early stopping should at least be 1, but got {} instead."
            )
            self.logger.error(msg.format(early_stopping_rounds))
            raise ValueError(msg.format(early_stopping_rounds))

        if not 0 < self.shrinkage_rate <= 1:
            msg = (
                "The shrinkage rate should be in the range (0, 1], but got"
                " {} instead."
            )
            self.logger.error(msg.format(self.shrinkage_rate))
            raise ValueError(msg.format(self.shrinkage_rate))

    @abc.abstractmethod
    def _handle_early_stopping(self, test_loader, est_idx):
        """Decide whether to trigger the internal counter on early stopping."""

    def _staged_forward(self, est_idx, *x):
        """
        Return the accumulated outputs from the first `est_idx+1` base
        estimators.
        """
        if est_idx >= self.n_estimators:
            msg = (
                "est_idx = {} should be an integer smaller than the"
                " number of base estimators = {}."
            )
            self.logger.error(msg.format(est_idx, self.n_estimators))
            raise ValueError(msg.format(est_idx, self.n_estimators))

        outputs = [
            estimator(*x) for estimator in self.estimators_[: est_idx + 1]
        ]
        out = op.sum_with_multiplicative(outputs, self.shrinkage_rate)

        return out

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
        self._validate_parameters(epochs, log_interval, early_stopping_rounds)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Utils
        criterion = (
            nn.MSELoss(reduction="sum") if use_reduction_sum else nn.MSELoss()
        )
        n_counter = 0  # a counter on early stopping

        for est_idx, estimator in enumerate(self.estimators_):

            # Initialize a optimizer and scheduler for each base estimator to
            # avoid unexpected dependencies.
            learner_optimizer = set_module.set_optimizer(
                estimator, self.optimizer_name, **self.optimizer_args
            )

            if self.use_scheduler_:
                learner_scheduler = set_module.set_scheduler(
                    learner_optimizer,
                    self.scheduler_name,
                    **self.scheduler_args  # noqa: E501
                )

            # Training loop
            estimator.train()
            total_iters = 0
            for epoch in range(epochs):
                for batch_idx, elem in enumerate(train_loader):

                    data, target = io.split_data_target(elem, self.device)

                    # Compute the learning target of the current estimator
                    residual = self._pseudo_residual(est_idx, target, *data)

                    output = estimator(*data)
                    loss = criterion(output, residual)

                    learner_optimizer.zero_grad()
                    loss.backward()
                    learner_optimizer.step()

                    # Print training status
                    if batch_idx % log_interval == 0:
                        msg = (
                            "Estimator: {:03d} | Epoch: {:03d} | Batch:"
                            " {:03d} | RegLoss: {:.5f}"
                        )
                        self.logger.info(
                            msg.format(est_idx, epoch, batch_idx, loss)
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "gradient_boosting/Est_{}/Train_Loss".format(
                                    est_idx
                                ),
                                loss,
                                total_iters,
                            )
                    total_iters += 1

                if self.use_scheduler_:
                    learner_scheduler.step()

            # Validation
            if test_loader:
                flag = self._handle_early_stopping(test_loader, est_idx)

                if flag:
                    n_counter += 1
                    msg = "Early stopping counter: {} out of {}"
                    self.logger.info(
                        msg.format(n_counter, early_stopping_rounds)
                    )

                    if n_counter == early_stopping_rounds:
                        msg = "Handling early stopping..."
                        self.logger.info(msg)

                        # Early stopping
                        offset = est_idx - n_counter
                        self.estimators_ = self.estimators_[: offset + 1]
                        self.n_estimators = len(self.estimators_)
                        break
                else:
                    # Reset the counter if the performance improves
                    n_counter = 0

        # Post-processing
        msg = "The optimal number of base estimators: {}"
        self.logger.info(msg.format(len(self.estimators_)))
        if save_model:
            io.save(self, save_dir, self.logger)


@_gradient_boosting_model_doc(
    """Implementation on the GradientBoostingClassifier.""", "model"
)
class GradientBoostingClassifier(_BaseGradientBoosting, BaseClassifier):
    def _pseudo_residual(self, est_idx, y, *x):
        """Compute pseudo residuals in classification."""
        output = torch.zeros(y.size(0), self.n_outputs).to(self.device)

        # Before fitting the first estimator, we simply assume that GBM
        # outputs 0 for any input (i.e., a null output).
        if est_idx > 0:
            results = [
                estimator(*x) for estimator in self.estimators_[:est_idx]
            ]
            output += op.sum_with_multiplicative(results, self.shrinkage_rate)
        pseudo_residual = op.pseudo_residual_classification(
            y, output, self.n_outputs
        )

        return pseudo_residual

    def _handle_early_stopping(self, test_loader, est_idx):
        # Compute the validation accuracy of base estimators fitted so far
        self.eval()
        correct = 0
        total = 0
        flag = False
        with torch.no_grad():
            for _, elem in enumerate(test_loader):
                data, target = io.split_data_target(elem, self.device)
                output = F.softmax(self._staged_forward(est_idx, *data), dim=1)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        acc = 100 * correct / total

        if est_idx == 0:
            self.best_acc = acc
        else:
            if acc > self.best_acc:
                self.best_acc = acc
            else:
                flag = True

        msg = "Validation Acc: {:.3f} % | Historical Best: {:.3f} %"
        self.logger.info(msg.format(acc, self.best_acc))
        if self.tb_logger:
            self.tb_logger.add_scalar(
                "gradient_boosting/Validation_Acc", acc, est_idx
            )

        return flag

    @torchensemble_model_doc(
        """Set the attributes on optimizer for GradientBoostingClassifier.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for GradientBoostingClassifier.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @_gradient_boosting_model_doc(
        """Implementation on the training stage of GradientBoostingClassifier.""",  # noqa: E501
        "fit",
    )
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
        self._criterion = nn.CrossEntropyLoss()
        super().fit(
            train_loader=train_loader,
            epochs=epochs,
            use_reduction_sum=use_reduction_sum,
            log_interval=log_interval,
            test_loader=test_loader,
            early_stopping_rounds=early_stopping_rounds,
            save_model=save_model,
            save_dir=save_dir,
        )

    @torchensemble_model_doc(
        """Implementation on the data forwarding in GradientBoostingClassifier.""",  # noqa: E501
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


@_gradient_boosting_model_doc(
    """Implementation on the GradientBoostingRegressor.""", "model"
)
class GradientBoostingRegressor(_BaseGradientBoosting, BaseRegressor):
    def _pseudo_residual(self, est_idx, y, *x):
        """Compute pseudo residuals in regression."""
        output = torch.zeros_like(y).to(self.device)

        if est_idx > 0:
            results = [
                estimator(*x) for estimator in self.estimators_[:est_idx]
            ]
            output = op.sum_with_multiplicative(results, self.shrinkage_rate)
        pseudo_residual = op.pseudo_residual_regression(y, output)

        return pseudo_residual

    def _handle_early_stopping(self, test_loader, est_idx):
        # Compute the validation MSE of base estimators fitted so far
        self.eval()
        mse = 0.0
        flag = False
        criterion = nn.MSELoss()
        with torch.no_grad():
            for _, elem in enumerate(test_loader):
                data, target = io.split_data_target(elem, self.device)
                output = self._staged_forward(est_idx, *data)
                mse += criterion(output, target)
        mse /= len(test_loader)

        if est_idx == 0:
            self.best_mse = mse
        else:
            assert hasattr(self, "best_mse")
            if mse < self.best_mse:
                self.best_mse = mse
            else:
                flag = True

        msg = "Validation MSE: {:.5f} | Historical Best: {:.5f}"
        self.logger.info(msg.format(mse, self.best_mse))
        if self.tb_logger:
            self.tb_logger.add_scalar(
                "gradient_boosting/Validation_MSE", mse, est_idx
            )

        return flag

    @torchensemble_model_doc(
        """Set the attributes on optimizer for GradientBoostingRegressor.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for GradientBoostingRegressor.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @_gradient_boosting_model_doc(
        """Implementation on the training stage of GradientBoostingRegressor.""",  # noqa: E501
        "fit",
    )
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
        self._criterion = nn.MSELoss()
        super().fit(
            train_loader=train_loader,
            epochs=epochs,
            use_reduction_sum=use_reduction_sum,
            log_interval=log_interval,
            test_loader=test_loader,
            early_stopping_rounds=early_stopping_rounds,
            save_model=save_model,
            save_dir=save_dir,
        )

    @torchensemble_model_doc(
        """Implementation on the data forwarding in GradientBoostingRegressor.""",  # noqa: E501
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
