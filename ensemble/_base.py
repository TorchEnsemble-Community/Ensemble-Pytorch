import abc
import torch
import torch.nn as nn


class BaseModule(abc.ABC, nn.Module):
    """
      Base class for ensemble methods.
      
      Warning: This class cannot be used directly. Use derived classes instead.
    """
    
    def __init__(self, 
                 estimator, 
                 n_estimators, 
                 output_dim, 
                 lr, 
                 weight_decay, 
                 epochs, 
                 cuda=True, 
                 log_interval=100):
        """
        Parameters
        ----------
        estimator : torch.nn.Module
            The base estimator class inherited from `torch.nn.Module`. Examples
            are available in the folder named `model`.
        n_estimators : int
            The number of base estimators in the ensemble model. With more base 
            estimators added, the performance of the entire ensemble model is 
            expected to improve.
        output_dim : int
            The output dimension of the model. For instance, for multi-class
            classification problem with K classes, it is set to `K`. For 
            univariate regression problem, it is set to `1`.
        lr : float
            The learning rate of the optimizer.
        weight_decay : float
            The weight decay of model parameters.
        epochs : int
            The number of training epochs.
        cuda : bool, default=True
            When set to `True`, use GPU to train and evaluate the model. When 
            set to `False`, the model is trained using CPU.
        log_interval : int, default=100
            The number of batches to wait before printing the training status,
            including information such as current batch, current epoch, current 
            training loss, and many more.
        
        Attributes
        ----------
        estimators_ : nn.ModuleList()
            A list that stores all base estimators.
        
        """
        super(BaseModule, self).__init__()
        
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.output_dim = output_dim
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        self.log_interval = log_interval
        self.device = torch.device('cuda' if cuda else 'cpu')
        
        # Initialize base estimators
        self.estimators_ = nn.ModuleList()
        for _ in range(self.n_estimators):
            self.estimators_.append(
                estimator(output_dim=output_dim).to(self.device))
        
        # Initialize a global optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                          lr=lr, 
                                          weight_decay=weight_decay)
    
    def __str__(self):
        msg = '==============================\n'
        msg += '{:<20}: {}\n'.format('Base Estimator', 
                                     self.estimator.__name__)
        msg += '{:<20}: {}\n'.format('n_estimators', self.n_estimators)
        msg += '{:<20}: {:.5f}\n'.format('Learning Rate', self.lr)
        msg += '{:<20}: {:.5f}\n'.format('Weight Decay', self.weight_decay)
        msg += '{:<20}: {}\n'.format('n_epochs', self.epochs)
        msg += '==============================\n'
        
        return msg
    
    def __repr__(self):
        return self.__str__()
    
    def _validate_parameters(self):
        
        if not self.n_estimators > 0:
            msg = ('The number of base estimators = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.n_estimators))
        
        if not self.output_dim > 0:
            msg = 'The output dimension = {} should not be strictly positive.'
            raise ValueError(msg.format(self.output_dim))
        
        if not self.lr > 0:
            msg = ('The learning rate of optimizer = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.lr))
        
        if not self.weight_decay >= 0:
            msg = 'The weight decay of parameters = {} should not be negative.'
            raise ValueError(msg.format(self.weight_decay))
        
        if not self.epochs > 0:
            msg = ('The number of training epochs = {} should be strictly'
                   ' positive.')
            raise ValueError(msg.format(self.epochs))
     
    @abc.abstractmethod
    def forward(self, X):
        """ Implementation on the data forwarding in the ensemble model. Notice
            that the input `X` should be a batch of data instead of a standalone
            data loader that contains many batches.
        """
    
    @abc.abstractmethod
    def fit(self, train_loader):
        """ Implementation on the training stage of the ensemble model.
        """
    
    @abc.abstractmethod
    def predict(self, test_loader):
        """ Implementation on the evaluating stage of the ensemble model.
        """
