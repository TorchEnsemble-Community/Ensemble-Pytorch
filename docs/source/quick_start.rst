Get started
===========

Install from PyPI
-----------------

Ensemble-PyTorch is available at PyPI, you can install it with the following command:

.. code-block:: none

    pip install torchensemble

Ensemble-PyTorch is designed to be portable and has very small package dependencies. We recommed to use the `Anaconda <https://www.anaconda.com/>`__ environment in combination with PyTorch installed from ``conda install pytorch``.

Define the Base Estimator
-------------------------

Since Ensemble-PyTorch uses ensemble methods to improve the performance, one key input argument is your customized model as the base estimator. Same as PyTorch, your model class should inherit from ``torch.nn.Module``, and it should at least implement two methods:

* ``__init__``: Instantiate sub-modules used in your model and assign them as member variables.
* ``forward``: Define the forward process of your model.

For example, the code snippet below defines a multi-layered perceptron (MLP) with the structure ``Input(90)-128-128-Output(1)``:

.. code-block:: python

    import torch.nn as nn
    from torch.nn import functional as F

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()

            self.linear1 = nn.Linear(90, 128)
            self.linear2 = nn.Linear(128, 128)
            self.linear3 = nn.Linear(128, 1)

        def forward(self, X):
            X = X.view(X.size()[0], -1)

            output = F.relu(self.linear1(X))
            output = F.dropout(output)
            output = F.relu(self.linear2(output))
            output = self.linear3(output)

            return output


Choose the Ensemble Wrapper
---------------------------

After implementing the model, we can then wrap it using one of the emsemble wrappers in Ensemble-PyTorch. Different wrappers have very similar APIs, take the ``VotingClassifier`` as an example:

.. code-block:: python

    from torchensemble.voting import VotingClassifier

    model = VotingClassifier(
        estimator=MLP,
        n_estimators=10,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs
    )

The meaning of different input arguments is listed as follow:

* ``estimator``: The class of your model, used in instantiate the base estimator in ensemble learning.
* ``n_estimators``: The number of base estimators.
* ``lr``: The learning rate of the internal Adam optimizer.
* ``weight_decay``: The weight decay of the internal Adam optimizer.
* ``epochs``: The number of training epochs.

.. note::
    The design on APIs is still on-going, and more options well be added latter.

Train and Evaluate
------------------

Ensemble-PyTorch provides Scikit-Learn APIs on the training and evaluating stage of the entire model:

.. code-block:: python

    # Training
    model.fit(train_loader)

    # Evaluating
    accuracy = model.predict(test_loader)

In the code snippet above, ``train_loader`` and ``test_loader`` is the PyTorch ``DataLoader`` wrapper on your own dataset.