.. image:: ./_images/badge.svg
   :align: center
   :width: 400

Ensemble PyTorch Documentation
==============================

Ensemble PyTorch is a unified ensemble framework for PyTorch to easily improve the performance and robustness of your deep learning model. It provides:

* Easy ways to improve the performance and robustness of your deep learning model.
* Easy-to-use APIs on training and evaluating the ensemble.
* High training efficiency with parallelization.

Guidepost
---------

* To get started, please refer to `Quick Start <./quick_start.html>`__;
* To learn more about ensemble methods supported, please refer to `Introduction <./introduction.html>`__;
* If you are confused on which ensemble method to use, our `experiments <./experiment.html>`__ and the instructions in `guidance <./guide.html>`__ may be helpful.

Example
-------

.. code:: python

  from torchensemble import VotingClassifier  # voting is a classic ensemble strategy

  # Load data
  train_loader = DataLoader(...)
  test_loader = DataLoader(...)

  # Define the ensemble
  ensemble = VotingClassifier(
      estimator=base_estimator,               # here is your deep learning model
      n_estimators=10,                        # number of base estimators
  )
  # Set the criterion
  criterion = nn.CrossEntropyLoss()           # training objective
  ensemble.set_criterion(criterion)

  # Set the optimizer
  ensemble.set_optimizer(
      "Adam",                                 # type of parameter optimizer
      lr=learning_rate,                       # learning rate of parameter optimizer
      weight_decay=weight_decay,              # weight decay of parameter optimizer
  )
  
  # Set the learning rate scheduler
  ensemble.set_scheduler(
      "CosineAnnealingLR",                    # type of learning rate scheduler
      T_max=epochs,                           # additional arguments on the scheduler
  )

  # Train the ensemble
  ensemble.fit(
      train_loader,
      epochs=epochs,                          # number of training epochs
  )

  # Evaluate the ensemble
  acc = ensemble.predict(test_loader)         # testing accuracy

Content
-------

.. toctree::
  :maxdepth: 1
  :caption: For Users

  Quick Start <quick_start>
  Introduction <introduction>
  Guidance <guide>
  Experiment <experiment>
  API Reference <parameters>
  Advanced Usage <advanced>

.. toctree::
  :maxdepth: 1
  :caption: For Developers

  Changelog <changelog>
  Roadmap <roadmap>
  Contributors <contributors>
  Code of Conduct <code_of_conduct>
