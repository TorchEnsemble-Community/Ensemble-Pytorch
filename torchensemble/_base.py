import abc
import torch
import torch.nn as nn


class BaseModule(abc.ABC, nn.Module):
    """Base class for ensemble methods.

    WARNING: This class cannot be used directly.
    Use the derived classes instead.
    """
    def __init__(self,
                 estimator,
                 n_estimators,
                 estimator_args=None,
                 cuda=True,
                 n_jobs=None):
        """
        Parameters
        ----------
        estimator : torch.nn.Module
            The class of base estimator inherited from ``torch.nn.Module``.
        n_estimators : int
            The number of base estimators in the ensemble.
        estimator_args : dict, default=None
            The dictionary of parameters used to instantiate base estimators.
        cuda : bool, default=True
            - If ``True``, use GPU to train and evaluate the ensemble.
            - If ``False``, use CPU to train and evaluate the ensemble.
        n_jobs : int, default=None
            The number of workers for training the ensemble. This
            argument is used for parallel ensemble methods such as
            :mod:`voting` and :mod:`bagging`. Setting it to an integer larger
            than ``1`` enables more than one base estimators to be jointly
            trained.

        Attributes
        ----------
        estimators_ : torch.nn.ModuleList
            An internal container that stores all base estimators.
        """
        super(BaseModule, self).__init__()

        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args
        self.device = torch.device("cuda" if cuda else "cpu")
        self.n_jobs = n_jobs

        self.estimators_ = nn.ModuleList()  # internal container

    def __len__(self):
        """Return the number of base estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index"th base estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over base estimators in the ensemble."""
        return iter(self.estimators_)

    def _decide_n_outputs(self, train_loader, is_classification):
        """Decide the number of outputs according to the train_loader."""
        n_outputs = None
        # For Classification: n_outputs = # classes
        if is_classification:
            if hasattr(train_loader.dataset, "classes"):
                n_outputs = len(train_loader.dataset.classes)
            # Infer from the dataloader
            else:
                labels = []
                for _, (_, target) in enumerate(train_loader):
                    labels.append(target)
                labels = torch.unique(torch.cat(labels))
                n_outputs = labels.size()[0]
        # For Regression: n_outputs = # target dimensions
        else:
            for _, (_, target) in enumerate(train_loader):
                n_outputs = target.size()[1]
                break
        return n_outputs

    def _make_estimator(self):
        """Make and configure a copy of the `base_estimator_`."""
        if self.estimator_args is None:
            estimator = self.base_estimator_()
        else:
            estimator = self.base_estimator_(**self.estimator_args)
        return estimator.to(self.device)

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

    @abc.abstractmethod
    def forward(self, X):
        """
        Implementation on the data forwarding in the ensemble. Notice
        that the input ``X`` should be a data batch instead of a standalone
        data loader that contains all data batches.
        """

    @abc.abstractmethod
    def fit(self,
            train_loader,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            optimizer="Adam",
            log_interval=100):
        """
        Implementation on the training stage of the ensemble.
        """

    @abc.abstractmethod
    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of the ensemble.
        """
