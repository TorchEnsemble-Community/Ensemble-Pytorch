.. image:: ./_images/badge.png
   :align: center
   :width: 400

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

Ensemble PyTorch Documentation
==============================

.. rst-class:: center

| |:homes:| `GitHub <https://github.com/xuyxu/Ensemble-Pytorch>`__ | |:book:| `ReadtheDocs <https://readthedocs.org/projects/ensemble-pytorch/>`__ | |:hammer_and_wrench:| `Codecov <https://codecov.io/gh/xuyxu/Ensemble-Pytorch>`__
|

**Ensemble-PyTorch** implements a collection of ensemble methods in PyTorch. It provides:

* |:arrow_up_small:| Easy ways to improve the performance and robustness of your deep learning model.
* |:eyes:| Easy-to-use APIs on training and evaluating the ensemble.
* |:zap:| High training efficiency with parallelization.

| This package is under active development. Please feel free to open an `issue <https://github.com/xuyxu/Ensemble-Pytorch/issues>`__ if your have any problem. In addition, any feature request or `pull request <https://github.com/xuyxu/Ensemble-Pytorch/pulls>`__ would be highly welcomed.

Guidepost
---------

* To get started, please refer to `Quick Start <./quick_start.html>`__;
* To learn more about ensemble methods supported, please refer to `Introduction <./introduction.html>`__;
* If you are confused on which ensemble method to use, instructions in `Guidance <./guide.html>`__ may be helpful.

Minimal Example on How to Use
-----------------------------

.. code:: python

    from torchensemble import ensemble_method           # import ensemble (e.g., VotingClassifier)

    # Load your dataset
    train_loader = DataLoader(...)
    test_loader = DataLoader(...)

    # Define the ensemble
    model = ensemble_method(estimator=base_estimator,   # your deep learning model
                            n_estimators=10)            # the number of base estimators

    # Set the optimizer
    model.set_optimizer("Adam",                         # parameter optimizer
                        lr=learning_rate,               # learning rate of the optimizer
                        weight_decay=weight_decay)      # weight decay of the optimizer

    # Train
    model.fit(train_loader,
              epochs=epochs)                            # the number of training epochs

    # Evaluate
    accuracy = model.predict(test_loader)               # evaluate the ensemble

Content
-------

.. toctree::
  :maxdepth: 2

   Quick Start <quick_start>
   Introduction <introduction>
   Guidance <guide>
   Experiment <experiment>
   API Reference <parameters>
   Changelog <changelog>
   Contributors <contributors>
   Roadmap <roadmap>
