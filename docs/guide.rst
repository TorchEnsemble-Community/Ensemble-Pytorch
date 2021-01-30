Guidance
========

This page provides useful instructions on how to choose the appropriate ensemble method for your deep learning model.

Check Your Model
----------------

A good rule-of-thumb is to check the performance of your deep learning model first. Below are two important aspects that you should pay attention to:

* What is the final performance of your model ?
* Does your model suffer from the over-fitting problem ?

To check these two aspects, it is recommended to evaluate the performance of your model on the :obj:`test_loader` after each training epoch.

.. tip::
    Using Ensemble-PyTorch, you can pass your model to the :class:`Fusion` or :class:`Voting` with the argument ``n_estimators`` set to ``1``. The behavior of the ensemble should be the same as a single model.

If the performance of your model is relatively good, for example, the testing accuracy of your LeNet-5 CNN model on MNIST is over 99%, the conclusion on the first point is that it is not likely that your model suffers from the under-fitting problem. You could skip the section :ref:`under_fit`.

.. _under_fit:

Under-fit
---------

If the performance of your model is unsatisfactory, you can try out the :class:`Gradient Boosting` related ensemble methods. :class:`Gradient Boosting` focuses on reducing the bias term from the perspective of `Bias-Variance Decomposition <https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff>`__, it usually works well when your deep learning model is a weak learner on the dataset.

Below is the pros and cons on using :class:`Gradient Boosting`:

* Pros:
    - You can have a much higher improvements than using other ensemble methods
* Cons:
    - Relatively longer training time
    - Suffer from the over-fitting problem if the value of ``n_estimators`` is large

.. tip::
    :class:`Gradient Boosting` in Ensemble-PyTorch supports the early stopping to alleviate the over-fitting. To use early stopping, you need to set the input argument ``test_loader`` and ``early_stopping_rounds`` when calling the :meth:`fit` function of :class:`Gradient Boosting`. In additional, using a small ``shrinkage_rate`` when declaring the model also helps to alleviate the over-fitting problem.

.. _over_fit:

Over-fit
--------

Large Training Costs
--------------------

Training an ensemble of large deep learning models could take prohibitively long time and easily run out of the memory. If you are suffering from large training costs when using Ensemble-PyTorch, the recommended ensemble method would be :class:`Snapshot Ensemble`. The training costs on :class:`Snapshot Ensemble` are approximately the same as that on training a single base estimator. Please refer to the related section in `Introduction <./introduction.html>`__ for details on :class:`Snapshot Ensemble`.

However, :class:`Snapshot Ensemble` does not work well across all deep learning models. To reduce the costs on using other parallel ensemble methods (i.e., :class:`Voting`, :class:`Bagging`, :class:`Adversarial Training`), you can set ``n_jobs`` to ``None`` or ``1``, which disables the parallelization conducted internally.
