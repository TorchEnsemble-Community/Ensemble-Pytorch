.. image:: ./docs/_images/badge_small.png

|github|_ |readthedocs|_ |codecov|_ |python|_ |pypi|_ |license|_

.. |github| image:: https://github.com/xuyxu/Ensemble-Pytorch/workflows/torchensemble-CI/badge.svg
.. _github: https://github.com/xuyxu/Ensemble-Pytorch/actions

.. |readthedocs| image:: https://readthedocs.org/projects/ensemble-pytorch/badge/?version=latest
.. _readthedocs: https://ensemble-pytorch.readthedocs.io/en/latest/index.html

.. |codecov| image:: https://codecov.io/gh/xuyxu/Ensemble-Pytorch/branch/master/graph/badge.svg?token=2FXCFRIDTV
.. _codecov: https://codecov.io/gh/xuyxu/Ensemble-Pytorch

.. |python| image:: https://img.shields.io/badge/python-3.6+-blue?logo=python
.. _python: https://www.python.org/

.. |pypi| image:: https://img.shields.io/pypi/v/torchensemble
.. _pypi: https://pypi.org/project/torchensemble/

.. |license| image:: https://img.shields.io/github/license/xuyxu/Ensemble-Pytorch
.. _license: https://github.com/xuyxu/Ensemble-Pytorch/blob/master/LICENSE

Ensemble PyTorch
================

Ensemble PyTorch is a unified ensemble framework for PyTorch to improve the performance and robustness of your deep learning model. Please refer to our `documentation <https://ensemble-pytorch.readthedocs.io/>`__ for details.

Installation
------------

Stable Version
~~~~~~~~~~~~~~

The stable version is available at `PyPI <https://pypi.org/project/torchensemble/>`__. You can install it using:

.. code:: bash

   $ pip install torchensemble

Latest Version
~~~~~~~~~~~~~~

To use the latest version, you need to install the package from source:

.. code:: bash

    $ git clone https://github.com/xuyxu/Ensemble-Pytorch.git
    $ cd Ensemble-Pytorch
    $ pip install -r requirements.txt (Optional)
    $ python setup.py install

Minimal Example on How to Use
-----------------------------

.. code:: python

    from torchensemble import VotingClassifier             # a classic ensemble method

    # Load your data
    train_loader = DataLoader(...)
    test_loader = DataLoader(...)

    # Define the ensemble
    model = VotingClassifier(estimator=base_estimator,     # your deep learning model
                             n_estimators=10)              # the number of base estimators

    # Set the optimizer
    model.set_optimizer("Adam",                            # parameter optimizer
                        lr=learning_rate,                  # learning rate of the optimizer
                        weight_decay=weight_decay)         # weight decay of the optimizer

    # Set the scheduler
    model.set_scheduler("CosineAnnealingLR", T_max=epochs) # optional

    # Train
    model.fit(train_loader,
              epochs=epochs)                               # the number of training epochs

    # Evaluate
    acc = model.predict(test_loader)                       # testing accuracy

Supported Ensemble
------------------

+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+
|    **Ensemble Name**    |  **Type**  |                                                                                                                             **Paper**                                                                                                                            |      **Repository**     |
+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+
|          Fusion         |    Mixed   |                                                                                                              Not listed for its Perceptual Intuition                                                                                                             |        fusion.py        |
+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+
|          Voting         |  Parallel  |                                                                                                              Not listed for its Perceptual Intuition                                                                                                             |        voting.py        |
+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+
|         Bagging         |  Parallel  |                                                                                       `Bagging Predictors <https://link.springer.com/content/pdf/10.1007/BF00058655.pdf>`__                                                                                      |        bagging.py       |
+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+
|    Gradient Boosting    | Sequential | `Greedy Function Approximation: A Gradient Boosting Machine <https://www.jstor.org/stable/pdf/2699986.pdf?casa_token=3fkT9safZHUAAAAA:HT_MeRk_xNsUZkOpbixOtXc950xnRSXNAyl7WjGZgjLtwBTAzZaQe2urnVyp5sK1dIXRL-9hVrdvjT-Ex_PEvov5tTyFg6wMaSbhCzkJRfUj4uBJ6l_PHA>`__ |   gradient_boosting.py  |
+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+
|    Snapshot Ensemble    | Sequential |                                                                                 `[ICLR'17] Snapshot Ensembles: Train 1, Get m For Free <https://arxiv.org/pdf/1704.00109.pdf>`__                                                                                 |   snapshot_ensemble.py  |
+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+
|   Adversarial Training  |  Parallel  |                                                                  `[NIPS'17] Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles <https://arxiv.org/pdf/1612.01474.pdf>`__                                                                 | adversarial_training.py |
+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+
| Fast Geometric Ensemble | Sequential |                                                                        `[NIPS'18] Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs <https://arxiv.org/pdf/1802.10026;Loss>`__                                                                       |    fast_geometric.py    |
+-------------------------+------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------+

Experiment
----------

Please refer to the `experiment part <https://ensemble-pytorch.readthedocs.io/en/stable/experiment.html>`__ of our documentation.

Package Dependency
------------------

-  scikit-learn>=0.23.0
-  torch>=1.4.0
-  torchvision>=0.2.2
