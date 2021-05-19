.. image:: ./docs/_images/badge_small.png

|github|_ |readthedocs|_ |codecov|_ |python|_ |pypi|_ |license|_ |commit_count|_

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

.. |commit_count| image:: https://img.shields.io/github/commits-since/xuyxu/Ensemble-PyTorch/latest
.. _commit_count: https://github.com/xuyxu/Ensemble-Pytorch

Ensemble PyTorch
================

A unified ensemble framework for pytorch_ to easily improve the performance and robustness of your deep learning model.

* `Document <https://ensemble-pytorch.readthedocs.io/>`__
* `Source Code <https://github.com/xuyxu/Ensemble-Pytorch>`__
* `Experiment <https://ensemble-pytorch.readthedocs.io/en/stable/experiment.html>`__

Installation
------------

Stable version:

.. code:: bash

    pip install torchensemble

Latest version (under development):

.. code:: bash

    pip install git+https://github.com/xuyxu/Ensemble-Pytorch

Example
-------

.. code:: python

    from torchensemble import VotingClassifier             # Voting is a classic ensemble strategy

    # Load data
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
    model.set_scheduler("CosineAnnealingLR", T_max=epochs) # (optional) learning rate scheduler

    # Train
    model.fit(train_loader,
              epochs=epochs)                               # the number of training epochs

    # Evaluate
    acc = model.predict(test_loader)                       # testing accuracy

Supported Ensemble
------------------

+------------------------------+------------+---------------------------+
|       **Ensemble Name**      |  **Type**  |      **Source Code**      |
+==============================+============+===========================+
|            Fusion            |    Mixed   |         fusion.py         |
+------------------------------+------------+---------------------------+
|          Voting [1]_         |  Parallel  |         voting.py         |
+------------------------------+------------+---------------------------+
|         Bagging [2]_         |  Parallel  |         bagging.py        |
+------------------------------+------------+---------------------------+
|    Gradient Boosting [3]_    | Sequential |    gradient_boosting.py   |
+------------------------------+------------+---------------------------+
|  Soft Gradient Boosting [7]_ |  Parallel  | soft_gradient_boosting.py |
+------------------------------+------------+---------------------------+
|    Snapshot Ensemble [4]_    | Sequential |    snapshot_ensemble.py   |
+------------------------------+------------+---------------------------+
|   Adversarial Training [5]_  |  Parallel  |  adversarial_training.py  |
+------------------------------+------------+---------------------------+
| Fast Geometric Ensemble [6]_ | Sequential |     fast_geometric.py     |
+------------------------------+------------+---------------------------+

Dependencies
------------

-  scikit-learn>=0.23.0
-  torch>=1.4.0
-  torchvision>=0.2.2

Reference
---------

.. [1] Zhou, Zhi-Hua. Ensemble Methods: Foundations and Algorithms. CRC press, 2012.

.. [2] Breiman, Leo. Bagging Predictors. Machine Learning (1996): 123-140.

.. [3] Friedman, Jerome H. Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics (2001): 1189-1232.

.. [4] Huang, Gao, et al. Snapshot Ensembles: Train 1, Get M For Free. ICLR, 2017.

.. [5] Lakshminarayanan, Balaji, et al. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NIPS, 2017.

.. [6] Garipov, Timur, et al. Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. NeurIPS, 2018.

.. [7] Feng, Ji, et al. Soft Gradient Boosting Machine. arXiv, 2020.

.. _pytorch: https://pytorch.org/

.. _pypi: https://pypi.org/project/torchensemble/