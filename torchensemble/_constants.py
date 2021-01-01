__model_doc = """
    Parameters
    ----------
    estimator : torch.nn.Module
        The class of base estimator inherited from :mod:`torch.nn.Module`.
    n_estimators : int
        The number of base estimators in the ensemble.
    estimator_args : dict, default=None
        The dictionary of hyper-parameters used to instantiate base
        estimators.
    cuda : bool, default=True

        - If ``True``, use GPU to train and evaluate the ensemble.
        - If ``False``, use CPU to train and evaluate the ensemble.
    n_jobs : int, default=None
        The number of workers for training the ensemble. This
        argument is used for parallel ensemble methods such as
        :mod:`voting` and :mod:`bagging`. Setting it to an integer larger
        than ``1`` enables a total number of ``n_jobs`` base estimators
        to be trained simultaneously.
    verbose : int, default=1
        Control the level on printing logging information.

        - If ``0``, trigger the silent mode
        - If ``1``, basic logging information on the training and
          evaluating status is printed.
        - If ``> 1``, full logging information is printed.

    Attributes
    ----------
    estimators_ : torch.nn.ModuleList
        The internal container that stores all base estimators.
"""


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the training data.
    lr : float, default=1e-3
        The learning rate of the parameter optimizer.
    weight_decay : float, default=5e-4
        The weight decay of the parameter optimizer.
    epochs : int, default=100
        The number of training epochs.
    optimizer : {"SGD", "Adam", "RMSprop"}, default="Adam"
        The type of parameter optimizer.
    log_interval : int, default=100
        The number of batches to wait before printting the training status.
    test_loader : torch.utils.data.DataLoader, default=None
        A :mod:`DataLoader` container that contains the evaluating data.

        - If ``None``, no validation is conducted after each training
          epoch.
        - If not ``None``, the ensemble will be evaluated on this
          dataloader after each training epoch.
    save_model : bool, default=True
        Whether to save the model.

        - If test_loader is ``None``, the ensemble trained over ``epochs``
          will be saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
"""


__classification_forward_doc = """
    Parameters
    ----------
    X : tensor
        Input batch of data, which should be a valid input data batch for
        base estimators.

    Returns
    -------
    proba : tensor of shape (batch_size, n_classes)
        The predicted class distribution.
"""


__classification_predict_doc = """
    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the testing data.

    Returns
    -------
    accuracy : float
        The testing accuracy of the fitted ensemble on the ``test_loader``.
"""


__regression_forward_doc = """
    Parameters
    ----------
    X : tensor
        Input batch of data, which should be a valid input data batch for
        base estimators.

    Returns
    -------
    pred : tensor of shape (batch_size, n_outputs)
        The predicted values.
"""


__regression_predict_doc = """
    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the testing data.

    Returns
    -------
    mse : float
        The testing mean squared error of the fitted model on the
        ``test_loader``.
"""
