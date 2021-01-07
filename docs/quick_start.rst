Get started
===========

Install from Source
-------------------

You can install the latest version of Ensemble-PyTorch with the following command:

.. code-block:: bash

    git clone https://github.com/xuyxu/Ensemble-Pytorch.git
    cd Ensemble-Pytorch
    pip install -r requirements.txt  # Optional
    python setup.py install

Ensemble-PyTorch is designed to be portable and has very small package dependencies. It is recommended to use the Python environment and PyTorch installed from `Anaconda <https://www.anaconda.com/>`__. In this case, there is no need to run the third command in the code snippet above.

Define Your Base Estimator
--------------------------

Since Ensemble-PyTorch uses ensemble methods to improve the performance, a key input argument is your deep learning model as the base estimator. Same as PyTorch, the class of your model should inherit from ``torch.nn.Module``, and it should at least implements two methods:

* ``__init__``: Instantiate sub-modules in your model and assign them as member variables.
* ``forward``: Define the forward process of your model.

For example, the code snippet below defines a multi-layered perceptron (MLP) with the structure `Input(90) - 128 - 128 - Output(1)`:

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


Choose the Ensemble Method
--------------------------

After implementing the base estimator, you can then choose one of the ensemble methods implemented in Ensemble-PyTorch. They have very similar APIs, take the ``VotingClassifier`` as an example:

.. code-block:: python

    from torchensemble.voting import VotingClassifier

    model = VotingClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )

The meaning of different arguments is listed as follow:

* ``estimator``: The class of your model, used to instantiate base estimators in the ensemble.
* ``n_estimators``: The number of base estimators in the ensemble.
* ``cuda``: Whether to use GPU to train and evaluate the ensemble.

Train and Evaluate
------------------

Ensemble-PyTorch provides Scikit-Learn APIs on the training and evaluating stage of the ensemble:

.. code-block:: python

    # Training
    model.fit(train_loader=train_loader,  # training data
              lr=1e-3,                    # learning rate of the optimizer
              weight_decay=5e-4,          # weight decay of the optimizer
              epochs=100,                 # number of training epochs
              optimizer="Adam")           # optimizer type

    # Evaluating
    accuracy = model.predict(test_loader)

In the code snippet above, ``train_loader`` and ``test_loader`` is the PyTorch ``DataLoader`` wrapper on your own dataset. In addition,

* ``lr``: The learning rate of the internal parameter optimizer.
* ``weight_decay``: The weight decay of the internal parameter optimizer.
* ``epochs``: The number of training epochs.
* ``optimizer``: Specify the type of the optimizer.

Since ``VotingClassifier`` is used for the classification problem, the predict function returns the classification accuracy on the ``test_loader``.

What's next
-----------
* You can check `Introduction <./introduction.html>`__ for details on ensemble methods available in Ensemble-PyTorch.
* You can check `Parameters <./parameters.html>`__ for detailed API design on different ensemble methods.