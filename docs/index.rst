.. image:: ./_images/badge.svg
   :align: center
   :width: 400

Ensemble PyTorch Documentation
==============================

Ensemble PyTorch is a unified ensemble framework for PyTorch to easily improve the performance and robustness of your deep learning model. It provides:

* |:arrow_up_small:| Easy ways to improve the performance and robustness of your deep learning model.
* |:eyes:| Easy-to-use APIs on training and evaluating the ensemble.
* |:zap:| High training efficiency with parallelization.

Guidepost
---------

* To get started, please refer to `Quick Start <./quick_start.html>`__;
* To learn more about ensemble methods supported, please refer to `Introduction <./introduction.html>`__;
* If you are confused on which ensemble method to use, our `experiments <./experiment.html>`__ and the instructions in `guidance <./guide.html>`__ may be helpful.

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

Content
-------

.. toctree::
  :maxdepth: 1

   Quick Start <quick_start>
   Introduction <introduction>
   Guidance <guide>
   Experiment <experiment>
   API Reference <parameters>
   Changelog <changelog>
   Contributors <contributors>
   Roadmap <roadmap>
