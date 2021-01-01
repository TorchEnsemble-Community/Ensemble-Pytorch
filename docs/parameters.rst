Parameters
==========

This page gives the API reference of ``torchensemble``, please also refer to `Introduction <./introduction.html>`__ for details.

Fusion
------

In fusion-based ensemble methods, the predictions from all base estimators
are first aggregated as an average output. After then, the training loss is
computed based on this average output and the ground-truth. The training loss
is then back-propagated to all base estimators simultaneously.

FusionClassifier
****************

.. autoclass:: torchensemble.fusion.FusionClassifier
    :members:
    :inherited-members:

FusionRegressor
***************

.. autoclass:: torchensemble.fusion.FusionRegressor
    :members:
    :inherited-members:

Voting
------

In voting-based ensemble methods, each base estimator is trained
independently, and the final prediction takes the average over predictions
from all base estimators.

VotingClassifier
****************

.. autoclass:: torchensemble.voting.VotingClassifier
    :members:

VotingRegressor
***************

.. autoclass:: torchensemble.voting.VotingRegressor
    :members:
    :inherited-members:

Bagging
-------

In bagging-based ensemble methods, each base estimator is trained
independently. In addition, sampling with replacement is conducted on the
training data to further encourge the diversity between different base
estimators in the ensemble model.

BaggingClassifier
*****************

.. autoclass:: torchensemble.bagging.BaggingClassifier
    :members:
    :inherited-members:

BaggingRegressor
****************

.. autoclass:: torchensemble.bagging.BaggingRegressor
    :members:
    :inherited-members:

Gradient Boosting
-----------------

Gradient boosting is a classic sequential ensemble method. At each iteration,
the learning target of a new base estimator is to fit the pseudo residual
computed based on the ground truth and the output from base estimators
fitted before, using ordinary least square.

.. tip::
    The input argument ``shrinkage_rate`` in :mod:`gradient_boosting` is also known as learning rate in other gradient boosting libraries such as `XGBoost <https://xgboost.readthedocs.io/en/latest/>`__. However, its meaning is totally different from the meaning of learning rate in the context of parameter optimizer in deep learning.

GradientBoostingClassifier
**************************

.. autoclass:: torchensemble.gradient_boosting.GradientBoostingClassifier
    :members:
    :inherited-members:

GradientBoostingRegressor
*************************

.. autoclass:: torchensemble.gradient_boosting.GradientBoostingRegressor
    :members:
    :inherited-members: