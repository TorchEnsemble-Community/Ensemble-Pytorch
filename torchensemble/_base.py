import abc
import torch
import logging
import torch.nn as nn

from . import _constants as const


def torchensemble_model_doc(header, item):
    """
    Decorator on obtaining documentation for different ensemble models,
    this decorator is modified from sklearn.py in XGBoost.

    Parameters
    ----------
    header: string
       An introducion to the decorated class.
    item : string
       The type of the docstring item.
    """
    def get_doc(item):
        """Return the selected item"""
        __doc = {"model": const.__model_doc,
                 "fit": const.__fit_doc,
                 "set_optimizer": const.__set_optimizer_doc,
                 "classifier_forward": const.__classification_forward_doc,
                 "classifier_predict": const.__classification_predict_doc,
                 "regressor_forward": const.__regression_forward_doc,
                 "regressor_predict": const.__regression_predict_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + "\n\n"]
        doc.extend(get_doc(item))
        cls.__doc__ = "".join(doc)
        return cls
    return adddoc


class BaseModule(abc.ABC, nn.Module):
    """Base class for ensemble methods.

    WARNING: This class cannot be used directly.
    Please use the derived classes instead.
    """
    def __init__(self,
                 estimator,
                 n_estimators,
                 estimator_args=None,
                 cuda=True,
                 n_jobs=None):
        super(BaseModule, self).__init__()

        # Make sure estimator is not an instance
        if not isinstance(estimator, type):
            msg = ("The input argument `estimator` should be a class"
                   " inherited from nn.Module. Perhaps you have passed"
                   " an instance of that class into the ensemble.")
            raise RuntimeError(msg)

        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args
        self.device = torch.device("cuda" if cuda else "cpu")
        self.n_jobs = n_jobs
        self.logger = logging.getLogger()

        self.estimators_ = nn.ModuleList()

    def __len__(self):
        """
        Return the number of base estimators in the ensemble. The actual number
        of base estimators may not match `n_estimators` because of the early
        stopping stage implemented in several ensembles.
        """
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the `index`-th base estimator in the ensemble."""
        return self.estimators_[index]

    def _decide_n_outputs(self, train_loader, is_classification=True):
        """
        Decide the number of outputs according to the train_loader.

        - If `is_classification`, the number of outputs equals the number of
          distinct classes.
        - If not `is_classification`, the number of outputs equals the number
          of target dimensions (e.g., 1 in univariate regression).
        """
        # For Classification: n_outputs = n_classes
        if is_classification:
            if hasattr(train_loader.dataset, "classes"):
                n_outputs = len(train_loader.dataset.classes)
            # Infer `n_outputs` from the dataloader
            else:
                labels = []
                for _, (_, target) in enumerate(train_loader):
                    labels.append(target)
                labels = torch.unique(torch.cat(labels))
                n_outputs = labels.size(0)
        # For Regression: n_outputs = n_target_dimensions
        else:
            for _, (_, target) in enumerate(train_loader):
                if len(target.size()) == 1:
                    n_outputs = 1
                else:
                    n_outputs = target.size(1)
                break

        return n_outputs

    def _make_estimator(self):
        """Make and configure a copy of the `self.base_estimator_`."""
        if self.estimator_args is None:
            estimator = self.base_estimator_()
        else:
            estimator = self.base_estimator_(**self.estimator_args)

        return estimator.to(self.device)

    def _validate_parameters(self, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if not epochs > 0:
            msg = ("The number of training epochs = {} should be strictly"
                   " positive.")
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))

        if not log_interval > 0:
            msg = ("The number of batches to wait before printting the"
                   " training status should be strictly positive, but got {}"
                   " instead.")
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))

    @abc.abstractmethod
    def set_optimizer(self, optimizer_name, **kwargs):
        """
        Implementation on the process of setting the optimizer.
        """

    @abc.abstractmethod
    def forward(self, x):
        """
        Implementation on the data forwarding in the ensemble. Notice
        that the input ``x`` should be a data batch instead of a standalone
        data loader that contains many data batches.
        """

    @abc.abstractmethod
    def fit(self,
            train_loader,
            epochs=100,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):
        """
        Implementation on the training stage of the ensemble.
        """

    @abc.abstractmethod
    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of the ensemble.
        """
