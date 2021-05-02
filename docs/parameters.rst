Parameters
==========

This page provides the API reference of :mod:`torchensemble`.

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

FusionRegressor
***************

.. autoclass:: torchensemble.fusion.FusionRegressor
    :members:

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

Bagging
-------

In bagging-based ensemble methods, each base estimator is trained
independently. In addition, sampling with replacement is conducted on the
training data to further encourage the diversity between different base
estimators in the ensemble.

BaggingClassifier
*****************

.. autoclass:: torchensemble.bagging.BaggingClassifier
    :members:

BaggingRegressor
****************

.. autoclass:: torchensemble.bagging.BaggingRegressor
    :members:

Gradient Boosting
-----------------

Gradient boosting is a classic sequential ensemble method. At each iteration,
the learning target of a new base estimator is to fit the pseudo residual
computed based on the ground truth and the output from base estimators
fitted before, using ordinary least square.

.. tip::
    The input argument ``shrinkage_rate`` in :class:`gradient_boosting` is also known as learning rate in other gradient boosting libraries such as `XGBoost <https://xgboost.readthedocs.io/en/latest/>`__. However, its meaning is totally different from the meaning of learning rate in the context of parameter optimizer in deep learning.

GradientBoostingClassifier
**************************

.. autoclass:: torchensemble.gradient_boosting.GradientBoostingClassifier
    :members:

GradientBoostingRegressor
*************************

.. autoclass:: torchensemble.gradient_boosting.GradientBoostingRegressor
    :members:

Snapshot Ensemble
-----------------

Snapshot ensemble generates many base estimators by enforcing a base
estimator to converge to its local minima many times and save the
model parameters at that point as a snapshot. The final prediction takes
the average over predictions from all snapshot models.

Reference:
    G. Huang, Y.-X. Li, G. Pleiss et al., Snapshot Ensemble: Train 1, and
    M for free, ICLR, 2017.

SnapshotEnsembleClassifier
**************************

.. autoclass:: torchensemble.snapshot_ensemble.SnapshotEnsembleClassifier
    :members:

SnapshotEnsembleRegressor
*************************

.. autoclass:: torchensemble.snapshot_ensemble.SnapshotEnsembleRegressor
    :members:

Adversarial Training
--------------------

Adversarial training is able to improve the performance of an ensemble by
treating adversarial samples as the augmented training data. The fast
gradient sign method (FGSM) is used to generate adversarial samples.

Reference:
    B. Lakshminarayanan, A. Pritzel, C. Blundell., Simple and Scalable
    Predictive Uncertainty Estimation using Deep Ensembles, NIPS 2017.

.. warning::
    When your base estimator is under-fit on the dataset, it is not recommended to use the :mod:`AdversarialTrainingClassifier` or :mod:`AdversarialTrainingRegressor`, because they may deteriorate the performance further.

AdversarialTrainingClassifier
*****************************

.. autoclass:: torchensemble.adversarial_training.AdversarialTrainingClassifier
    :members:

AdversarialTrainingRegressor
*****************************

.. autoclass:: torchensemble.adversarial_training.AdversarialTrainingRegressor
    :members:

Fast Geometric Ensemble
-----------------------

Motivated by geometric insights on the loss surface of deep neural networks,
Fast Geometric Ensembling (FGE) is an efficient ensemble that uses a
customized learning rate scheduler to generate base estimators, similar to
snapshot ensemble.

Reference:
    T. Garipov, P. Izmailov, D. Podoprikhin et al., Loss Surfaces, Mode
    Connectivity, and Fast Ensembling of DNNs, NeurIPS, 2018.

FastGeometricClassifier
***********************

.. autoclass:: torchensemble.fast_geometric.FastGeometricClassifier
    :members:

FastGeometricRegressor
***********************

.. autoclass:: torchensemble.fast_geometric.FastGeometricRegressor
    :members:
