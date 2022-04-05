import abc
import copy
import torch
import logging
import warnings
import numpy as np
import torch.nn as nn

from . import _constants as const
from .utils.io import split_data_target
from .utils.logging import get_tb_logger


def torchensemble_model_doc(header="", item="model"):
    """
    A decorator on obtaining documentation for different methods in the
    ensemble. This decorator is modified from `sklearn.py` in XGBoost.

    Parameters
    ----------
    header: string
       Introduction to the decorated class or method.
    item : string
       Type of the docstring item.
    """

    def get_doc(item):
        """Return the selected item."""
        __doc = {
            "model": const.__model_doc,
            "seq_model": const.__seq_model_doc,
            "tree_ensmeble_model": const.__tree_ensemble_doc,
            "fit": const.__fit_doc,
            "predict": const.__predict_doc,
            "set_optimizer": const.__set_optimizer_doc,
            "set_scheduler": const.__set_scheduler_doc,
            "set_criterion": const.__set_criterion_doc,
            "classifier_forward": const.__classification_forward_doc,
            "classifier_evaluate": const.__classification_evaluate_doc,
            "regressor_forward": const.__regression_forward_doc,
            "regressor_evaluate": const.__regression_evaluate_doc,
        }
        return __doc[item]

    def adddoc(cls):
        doc = [header + "\n\n"]
        doc.extend(get_doc(item))
        cls.__doc__ = "".join(doc)
        return cls

    return adddoc


class BaseModule(nn.Module):
    """Base class for all ensembles.

    WARNING: This class cannot be used directly.
    Please use the derived classes instead.
    """

    def __init__(
        self,
        estimator,
        n_estimators,
        estimator_args=None,
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

        self.device = torch.device("cuda" if cuda else "cpu")
        self.n_jobs = n_jobs
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()

        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def __len__(self):
        """
        Return the number of base estimators in the ensemble. The real number
        of base estimators may not match `self.n_estimators` because of the
        early stopping stage in several ensembles such as Gradient Boosting.
        """
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the `index`-th base estimator in the ensemble."""
        return self.estimators_[index]

    @abc.abstractmethod
    def _decide_n_outputs(self, train_loader):
        """Decide the number of outputs according to the `train_loader`."""

    def _make_estimator(self):
        """Make and configure a copy of `self.base_estimator_`."""

        # Call `deepcopy` to make a base estimator
        if not isinstance(self.base_estimator_, type):
            estimator = copy.deepcopy(self.base_estimator_)
        # Call `__init__` to make a base estimator
        else:
            # Without params
            if self.estimator_args is None:
                estimator = self.base_estimator_()
            # With params
            else:
                estimator = self.base_estimator_(**self.estimator_args)

        return estimator.to(self.device)

    def _validate_parameters(self, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if not epochs > 0:
            msg = (
                "The number of training epochs should be strictly positive"
                ", but got {} instead."
            )
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))

        if not log_interval > 0:
            msg = (
                "The number of batches to wait before printing the"
                " training status should be strictly positive, but got {}"
                " instead."
            )
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))

    def set_criterion(self, criterion):
        """Set the training criterion."""
        self._criterion = criterion

    def set_optimizer(self, optimizer_name, **kwargs):
        """Set the parameter optimizer."""
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    def set_scheduler(self, scheduler_name, **kwargs):
        """Set the learning rate scheduler."""
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    @abc.abstractmethod
    def forward(self, *x):
        """
        Implementation on the data forwarding in the ensemble. Notice
        that the input ``x`` should be a data batch instead of a standalone
        data loader that contains many data batches.
        """

    @abc.abstractmethod
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        """
        Implementation on the training stage of the ensemble.
        """

    @torch.no_grad()
    def predict(self, *x):
        """Docstrings decorated by downstream ensembles."""
        self.eval()

        # Copy data
        x_device = []
        for data in x:
            if isinstance(data, torch.Tensor):
                x_device.append(data.to(self.device))
            elif isinstance(data, np.ndarray):
                x_device.append(torch.Tensor(data).to(self.device))
            else:
                msg = (
                    "The type of input X should be one of {{torch.Tensor,"
                    " np.ndarray}}."
                )
                raise ValueError(msg)

        pred = self.forward(*x_device)
        pred = pred.cpu()
        return pred


class BaseTreeEnsemble(BaseModule):
    def __init__(
        self,
        n_estimators,
        depth=5,
        lamda=1e-3,
        cuda=False,
        n_jobs=None,
    ):
        super(BaseModule, self).__init__()
        self.base_estimator_ = BaseTree
        self.n_estimators = n_estimators
        self.depth = depth
        self.lamda = lamda

        self.device = torch.device("cuda" if cuda else "cpu")
        self.n_jobs = n_jobs
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()

        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def _decidce_n_inputs(self, train_loader):
        """Decide the input dimension according to the `train_loader`."""
        for _, elem in enumerate(train_loader):
            data = elem[0]
            n_samples = data.size(0)
            data = data.view(n_samples, -1)
            return data.size(1)

    def _make_estimator(self):
        """Make and configure a soft decision tree."""
        estimator = BaseTree(
            input_dim=self.n_inputs,
            output_dim=self.n_outputs,
            depth=self.depth,
            lamda=self.lamda,
            cuda=self.device == torch.device("cuda"),
        )

        return estimator.to(self.device)


class BaseClassifier(BaseModule):
    """Base class for all ensemble classifiers.

    WARNING: This class cannot be used directly.
    Please use the derived classes instead.
    """

    def _decide_n_outputs(self, train_loader):
        """
        Decide the number of outputs according to the `train_loader`.
        The number of outputs equals the number of distinct classes for
        classifiers.
        """
        if hasattr(train_loader.dataset, "classes"):
            n_outputs = len(train_loader.dataset.classes)
        # Infer `n_outputs` from the dataloader
        else:
            labels = []
            for _, elem in enumerate(train_loader):
                _, target = split_data_target(elem, self.device)
                labels.append(target)
            labels = torch.unique(torch.cat(labels))
            n_outputs = labels.size(0)

        return n_outputs

    @torch.no_grad()
    def evaluate(self, test_loader, return_loss=False):
        """Docstrings decorated by downstream models."""
        self.eval()
        correct = 0
        total = 0
        loss = 0.0

        for _, elem in enumerate(test_loader):
            data, target = split_data_target(elem, self.device)
            output = self.forward(*data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            loss += self._criterion(output, target)

        acc = 100 * correct / total
        loss /= len(test_loader)

        if return_loss:
            return acc, float(loss)

        return acc


class BaseRegressor(BaseModule):
    """Base class for all ensemble regressors.

    WARNING: This class cannot be used directly.
    Please use the derived classes instead.
    """

    def _decide_n_outputs(self, train_loader):
        """
        Decide the number of outputs according to the `train_loader`.
        The number of outputs equals the number of target variables for
        regressors (e.g., `1` in univariate regression).
        """
        for _, elem in enumerate(train_loader):
            _, target = split_data_target(elem, self.device)
            if len(target.size()) == 1:
                n_outputs = 1  # univariate regression
            else:
                n_outputs = target.size(1)  # multivariate regression
            break

        return n_outputs

    @torch.no_grad()
    def evaluate(self, test_loader):
        """Docstrings decorated by downstream ensembles."""
        self.eval()
        loss = 0.0

        for _, elem in enumerate(test_loader):
            data, target = split_data_target(elem, self.device)
            output = self.forward(*data)
            loss += self._criterion(output, target)

        return float(loss) / len(test_loader)


class BaseTree(nn.Module):
    """Fast implementation of soft decision tree in PyTorch, copied from:
    `https://github.com/xuyxu/Soft-Decision-Tree/blob/master/SDT.py`
    """

    def __init__(self, input_dim, output_dim, depth=5, lamda=1e-3, cuda=False):
        super(BaseTree, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.depth = depth
        self.lamda = lamda
        self.device = torch.device("cuda" if cuda else "cpu")

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [
            self.lamda * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )

        self.leaf_nodes = nn.Linear(
            self.leaf_node_num_, self.output_dim, bias=False
        )

    def forward(self, X, is_training_data=False):
        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        """Implementation on the data forwarding process."""

        batch_size = X.size()[0]
        X = self._data_augment(X)

        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0).to(self.device)

        # Iterate through internal odes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """Compute the regularization term for internal nodes"""

        penalty = torch.tensor(0.0).to(self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0
            ) / torch.sum(_mu[:, node // 2], dim=0)

            coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = (
                "The tree depth should be strictly positive, but got {}"
                "instead."
            )
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = (
                "The coefficient of the regularization term should not be"
                " negative, but got {} instead."
            )
            raise ValueError(msg.format(self.lamda))
