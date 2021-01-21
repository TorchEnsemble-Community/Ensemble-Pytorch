.. image:: ./docs/_images/badge_small.png

|github|_ |readthedocs|_ |codecov|_ |python|_

.. |github| image:: https://github.com/xuyxu/Ensemble-Pytorch/workflows/torchensemble-CI/badge.svg
.. _github: https://github.com/xuyxu/Ensemble-Pytorch/actions

.. |readthedocs| image:: https://readthedocs.org/projects/ensemble-pytorch/badge/?version=latest
.. _readthedocs: https://ensemble-pytorch.readthedocs.io/en/latest/index.html

.. |codecov| image:: https://codecov.io/gh/xuyxu/Ensemble-Pytorch/branch/master/graph/badge.svg?token=2FXCFRIDTV
.. _codecov: https://codecov.io/gh/xuyxu/Ensemble-Pytorch

.. |python| image:: https://img.shields.io/badge/python-3.6+-blue?logo=python
.. _python: https://www.python.org/

Ensemble Pytorch
================

Implementation of ensemble methods in Pytorch to boost the performance of your model.

List of methods
---------------

+--------+----------------------+-------------------+
| **ID** |    **Method Name**   | **Ensemble Type** |
+--------+----------------------+-------------------+
|    1   |        Fusion        |       Mixed       |
+--------+----------------------+-------------------+
|    2   |        Voting        |      Parallel     |
+--------+----------------------+-------------------+
|    3   |        Bagging       |      Parallel     |
+--------+----------------------+-------------------+
|    4   |   Gradient Boosting  |     Sequential    |
+--------+----------------------+-------------------+
|    5   |   Snapshot Ensemble  |     Sequential    |
+--------+----------------------+-------------------+
|    6   | Adversarial Training |      Parallel     |
+--------+----------------------+-------------------+

News
----

-  A pre-released version of documentation is available at
   https://ensemble-pytorch.readthedocs.io/en/latest/.

Installation
------------

Installing Ensemble-Pytorch package is simple. Just clone this repo and
run ``setup.py``.

.. code:: bash

    $ git clone https://github.com/AaronX121/Ensemble-Pytorch.git
    $ cd Ensemble-Pytorch
    $ pip install -r requirements.txt
    $ python setup.py install

Minimal example on how to use
-----------------------------

.. code:: python

    from torchensemble import ensemble_method           # import ensemble method (e.g., VotingClassifier)

    # Define the ensemble
    model = ensemble_method(estimator=base_estimator,   # class of your base estimator
                            n_estimators=10,            # the number of base estimators              
                            cuda=True)                  # whether to use GPU            

    # Load data
    train_loader = DataLoader(...)
    test_loader = DataLoader(...)

    # Set the optimizer
    model.set_optimizer("Adam",                         # type of the parameter optimizer
                        lr=learning_rate,               # learning rate of the parameter optimizer
                        weight_decay=weight_decay)      # weight decay of the parameter optimizer

    # Train
    model.fit(train_loader,                             # training data
              epochs=epochs)                            # the number of training epochs

    # Evaluate
    accuracy = model.predict(test_loader)

Benchmarks
----------

-  **Classification on CIFAR-10**
-  The table below presents the classification accuracy of different
   ensemble classifiers on the testing data of **CIFAR-10**
-  Each classifier uses **10** LeNet-5 model (with RELU activation and
   Dropout) as the base estimators
-  Each base estimator is trained over **100** epochs, with batch size
   **128**, learning rate **1e-3**, and weight decay **5e-4**
-  Experiment results can be reproduced by running
   ``./examples/classification_cifar10_cnn.py``

+----------------------------------+---------------+-------------------+-------------------+
| Model Name                       | Params (MB)   | Testing Acc (%)   | Improvement (%)   |
+==================================+===============+===================+===================+
| **Single LeNet-5**               | 0.32          | 73.04             | 0                 |
+----------------------------------+---------------+-------------------+-------------------+
| **FusionClassifier**             | 3.17          | 78.75             | +5.71             |
+----------------------------------+---------------+-------------------+-------------------+
| **VotingClassifier**             | 3.17          | 80.08             | +7.04             |
+----------------------------------+---------------+-------------------+-------------------+
| **BaggingClassifier**            | 3.17          | 78.75             | +5.71             |
+----------------------------------+---------------+-------------------+-------------------+
| **GradientBoostingClassifier**   | 3.17          | 80.82             | **+7.78**         |
+----------------------------------+---------------+-------------------+-------------------+

-  **Regression on YearPredictionMSD**
-  The table below presents the mean squared error (MSE) of different
   ensemble regressors on the testing data of **YearPredictionMSD**
-  Each regressor uses **10** multi-layered perceptron (MLP) model (with
   RELU activation and Dropout) as the base estimators, and the network
   architecture is fixed as ``Input-128-128-Output``
-  Each base estimator is trained over **50** epochs, with batch size
   **256**, learning rate **1e-3**, and weight decay **5e-4**
-  Experiment results can be reproduced by running
   ``./examples/regression_YearPredictionMSD_mlp.py``

+---------------------------------+---------------+---------------+---------------+
| Model Name                      | Params (MB)   | Testing MSE   | Improvement   |
+=================================+===============+===============+===============+
| **Single MLP**                  | 0.11          | 0.83          |               |
+---------------------------------+---------------+---------------+---------------+
| **FusionRegressor**             | 1.08          | 0.73          | -0.10         |
+---------------------------------+---------------+---------------+---------------+
| **VotingRegressor**             | 1.08          | 0.69          | **-0.14**     |
+---------------------------------+---------------+---------------+---------------+
| **BaggingRegressor**            | 1.08          | 0.70          | -0.13         |
+---------------------------------+---------------+---------------+---------------+
| **GradientBoostingRegressor**   | 1.08          | 0.71          | -0.12         |
+---------------------------------+---------------+---------------+---------------+

Package dependencies
--------------------

-  joblib>=0.11
-  scikit-learn>=0.23.0
-  torch>=0.4.1
-  torchvision>=0.2.2
