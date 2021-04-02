Get started
===========

Install the Stable Version
--------------------------

You can use the stable version of Ensemble-PyTorch with the following command:

.. code-block:: bash

    $ pip install torchensemble

Ensemble-PyTorch is designed to be portable and has very small package dependencies. It is recommended to use the package environment and PyTorch installed from `Anaconda <https://www.anaconda.com/>`__.

Define Your Base Estimator
--------------------------

Since Ensemble-PyTorch uses different ensemble methods to improve the performance, a key input argument is your deep learning model, serving as the base estimator. Same as PyTorch, the class of your model should inherit from :mod:`torch.nn.Module`, and it should at least implement two methods:

* ``__init__``: Instantiate sub-modules in your model and assign them as the member variables.
* ``forward``: Define the forward process of your model.

For example, the code snippet below defines a multi-layered perceptron (MLP) of the structure: Input(784) - 128 - 128 - Output(10):

.. code-block:: python

    import torch.nn as nn
    from torch.nn import functional as F

    class MLP(nn.Module):

        def __init__(self):
            super(MLP, self).__init__()

            self.linear1 = nn.Linear(784, 128)
            self.linear2 = nn.Linear(128, 128)
            self.linear3 = nn.Linear(128, 10)

        def forward(self, X):
            X = X.view(X.size(0), -1)

            output = F.relu(self.linear1(X))
            output = F.dropout(output)
            output = F.relu(self.linear2(output))
            output = self.linear3(output)

            return output

Set the Logger
--------------

Ensemble-PyTorch uses a global logger to track and print the intermediate information. The code snippet below shows how to set up a logger:

.. code-block:: python

    from torchensemble.utils import set_logger

    logger = set_logger("classification_mnist_mlp")

With this logger, all intermediate information will be printed on the command line and saved to the specified text file: ``classification_mnist_mlp``. Besides, you can use tensorboard to have a better visualization result on training and evaluating the ensemble.

.. code-block:: bash

    tensorboard --logdir=logs/

The tensorboard feature can be disabled by passing ``use_tb_logger=False`` into the method :meth:`set_logger`.

Choose the Ensemble
-------------------

After defining the base estimator, we can then wrap it using one of ensemble methods available in Ensemble-PyTorch. Different methods have very similar APIs, take the ``VotingClassifier`` as an example:

.. code-block:: python

    from torchensemble import VotingClassifier

    model = VotingClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )

The meaning of different arguments is listed as follow:

* ``estimator``: The class of your model, used to instantiate base estimators in the ensemble.
* ``n_estimators``: The number of base estimators.
* ``cuda``: Specify whether to use GPU for training and evaluating the ensemble.

Set the Optimizer
-----------------

After creating the ensemble, another step before the training stage is to set the optimizer. Suppose that we are going to use the Adam optimizer with learning rate ``1e-3`` and weight decay ``5e-4`` to train the ensemble, this can be achieved by calling the ``set_optimizer`` method of the ensemble:

.. code-block:: python

    model.set_optimizer("Adam",             # optimizer name
                        lr=1e-3,            # learning rate of the optimizer
                        weight_decay=5e-4)  # weight decay of the optimizer

Notice that all arguments after the optimizer name (i.e., ``Adam``) should be in the form of keyword arguments. They be will directly delivered to the :mod:`torch.optim.Optimizer`.

Setting the scheduler for the ensemble is also supported in Ensemble-Pytorch, please refer to the ``set_scheduler`` method in `API Reference <./parameters.html>`__.

Train and Evaluate
------------------

Given the ensemble with the optimizer already set, Ensemble-PyTorch provides Scikit-Learn APIs on the training and evaluating stage of the ensemble:

.. code-block:: python

    # Training
    model.fit(train_loader=train_loader,  # training data
              epochs=100)                 # number of training epochs

    # Evaluating
    accuracy = model.predict(test_loader)

In the code snippet above, ``train_loader`` and ``test_loader`` is the PyTorch :mod:`DataLoader` object that contains your own dataset. In addition, ``epochs`` specify the number of training epochs. Since ``VotingClassifier`` is used for the classification, the ``predict`` function will return the classification accuracy on the ``test_loader``.

Notice that the ``test_loader`` can also be passed to ``fit``, under the case, the ensemble will be evaluated on the ``test_loader`` after each training epoch.

Example on MNIST
----------------

The script below shows a concrete example on using VotingClassifier with 10 MLPs for classification on the MNIST dataset.

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torchvision import datasets, transforms

    from torchensemble import VotingClassifier
    from torchensemble.utils.logging import set_logger

    # Define Your Base Estimator
    class MLP(nn.Module):

        def __init__(self):
            super(MLP, self).__init__()

            self.linear1 = nn.Linear(784, 128)
            self.linear2 = nn.Linear(128, 128)
            self.linear3 = nn.Linear(128, 10)

        def forward(self, X):
            X = X.view(X.size(0), -1)
            output = F.relu(self.linear1(X))
            output = F.dropout(output)
            output = F.relu(self.linear2(output))
            output = self.linear3(output)

            return output

    # Load MNIST dataset
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train = datasets.MNIST('../../Dataset', train=True, download=True, transform=transform)
    test = datasets.MNIST('../../Dataset', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

    # Set the Logger
    logger = set_logger("classification_mnist_mlp")

    # Set the model
    model = VotingClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    model.set_optimizer("Adam", lr=1e-3, weight_decay=5e-4)

    # Train and Evaluate
    model.fit(train_loader,
              epochs=50,
              test_loader=test_loader)

What's next
-----------
* You can check `Introduction <./introduction.html>`__ for details on ensemble methods available in Ensemble-PyTorch.
* You can check `API Reference <./parameters.html>`__ for detailed API design on ensemble methods.
