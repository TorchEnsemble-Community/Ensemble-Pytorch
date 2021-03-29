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


__seq_model_doc = """
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
    cuda : bool, default=True

        - If ``True``, use GPU to train and evaluate the ensemble.
        - If ``False``, use CPU to train and evaluate the ensemble.

    Attributes
    ----------
    estimators_ : torch.nn.ModuleList
        An internal container that stores all fitted base estimators.
"""


__set_optimizer_doc = """
    Parameters
    ----------
    optimizer_name : string
        The name of the optimizer, should be one of {``Adadelta``, ``Adagrad``,
        ``Adam``, ``AdamW``, ``Adamax``, ``ASGD``, ``RMSprop``, ``Rprop``,
        ``SGD``}.
    **kwargs : keyword arguments
        Keyword arguments on setting the optimizer, should be in the form:
        ``lr=1e-3, weight_decay=5e-4, ...``. These keyword arguments
        will be directly passed to :mod:`torch.optim.Optimizer`.
"""


__set_scheduler_doc = """
    Parameters
    ----------
    scheduler_name : string
        The name of the scheduler, should be one of {``LambdaLR``,
        ``MultiplicativeLR``, ``StepLR``, ``MultiStepLR``, ``ExponentialLR``,
        ``CosineAnnealingLR``, ``ReduceLROnPlateau``}.
    **kwargs : keyword arguments
        Keyword arguments on setting the scheduler. These keyword arguments
        will be directly passed to :mod:`torch.optim.lr_scheduler`.
"""


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A :mod:`torch.utils.data.DataLoader` container that contains the
        training data.
    epochs : int, default=100
        The number of training epochs.
    log_interval : int, default=100
        The number of batches to wait before logging the training status.
    test_loader : torch.utils.data.DataLoader, default=None
        A :mod:`torch.utils.data.DataLoader` container that contains the
        evaluating data.

        - If ``None``, no validation is conducted during the training
          stage.
        - If not ``None``, the ensemble will be evaluated on this
          dataloader after each training epoch.
    save_model : bool, default=True
        Specify whether to save the model parameters.

        - If test_loader is ``None``, the ensemble fully trained will be
          saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model parameters.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
    tb_logger : tensorboard.SummaryWriter, default=None
        Specify whether the data will be recorded by tensorboard writer

        - If ``None``, the data will not be recorded
        - If not ``None``, the data wiil be recorded by the given ``tb_logger``
"""


__classification_forward_doc = """
    Parameters
    ----------
    X : tensor
        An input batch of data, which should be a valid input data batch
        for base estimators in the ensemble.

    Returns
    -------
    proba : tensor of shape (batch_size, n_classes)
        The predicted class distribution.
"""


__classification_predict_doc = """
    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        A :mod:`torch.utils.data.DataLoader` container that contains the
        evaluating data.

    Returns
    -------
    accuracy : float
        The testing accuracy of the fitted ensemble on ``test_loader``.
"""


__regression_forward_doc = """
    Parameters
    ----------
    X : tensor
        An input batch of data, which should be a valid input data batch
        for base estimators in the ensemble.

    Returns
    -------
    pred : tensor of shape (batch_size, n_outputs)
        The predicted values.
"""


__regression_predict_doc = """
    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        A :mod:`torch.utils.data.DataLoader` container that contains the
        evaluating data.

    Returns
    -------
    mse : float
        The testing mean squared error (MSE) of the fitted ensemble on
        ``test_loader``.
"""
