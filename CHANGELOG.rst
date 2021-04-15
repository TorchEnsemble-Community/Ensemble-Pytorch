Changelog
=========

Ver 0.1.*
---------

* |Enhancement| Relax :mod:`tensorboard` as a soft dependency | `@xuyxu <https://github.com/xuyxu>`__
* |Enhancement| |API| Simplify the training workflow of :class:`FastGeometricClassifier` and :class:`FastGeometricRegressor` | `@xuyxu <https://github.com/xuyxu>`__
* |Feature| |API| Support TensorBoard logging in :meth:`set_logger` | `@zzzzwj <https://github.com/zzzzwj>`__
* |Enhancement| |API| Add ``use_reduction_sum`` parameter for :meth:`fit` of Gradient Boosting | `@xuyxu <https://github.com/xuyxu>`__
* |Feature| |API| Improve the functionality of :meth:`evaluate` and :meth:`predict` | `@xuyxu <https://github.com/xuyxu>`__
* |Feature| |API| Add :class:`FastGeometricClassifier` and :class:`FastGeometricRegressor` | `@xuyxu <https://github.com/xuyxu>`__
* |Enhancement| Add flexible instantiation of optimizers and schedulers | `@cspsampedro <https://github.com/cspsampedro>`__
* |Feature| |API| Add support on accepting instantiated base estimators as valid input | `@xuyxu <https://github.com/xuyxu>`__
* |Fix| Fix missing base estimators when calling :meth:`load()` for all ensembles | `@xuyxu <https://github.com/xuyxu>`__
* |Feature| |API| Add methods on model deserialization :meth:`load()` for all ensembles | `@mttgdd <https://github.com/mttgdd>`__

Beta
----

* |Feature| |API| Add :meth:`set_scheduler` for all ensembles | `@xuyxu <https://github.com/xuyxu>`__
* |MajorFeature| Add :class:`AdversarialTrainingClassifier` and :class:`AdversarialTrainingRegressor` | `@xuyxu <https://github.com/xuyxu>`__
* |MajorFeature| Add :class:`SnapshotEnsembleClassifier` and :class:`SnapshotEnsembleRegressor` | `@xuyxu <https://github.com/xuyxu>`__
* |Feature| |API| Add model validation and serialization | `@ozanpkr <https://github.com/ozanpkr>`__ and `@xuyxu <https://github.com/xuyxu>`__
* |Enhancement| Add CI and maintenance tools | `@xuyxu <https://github.com/xuyxu>`__
* |Enhancement| Add the code coverage on codecov | `@xuyxu <https://github.com/xuyxu>`__
* |Enhancement| Add the version numbers to requirements.txt | `@zackhardtoname <https://github.com/zackhardtoname>`__ and `@xuyxu <https://github.com/xuyxu>`__
* |Enhancement| Improve the logging module using :class:`logging` | `@zzzzwj <https://github.com/zzzzwj>`__
* |API| Remove the input argument ``output_dim`` from all methods | `@xuyxu <https://github.com/xuyxu>`__
* |API| Refactor the setup on optimizer into :meth:`set_optimizer` | `@xuyxu <https://github.com/xuyxu>`__
* |API| Refactor the codes on operating tensors into an independent module | `@zzzzwj <https://github.com/zzzzwj>`__
* |Fix| Fix the bug in logging module when using multi-processing | `@zzzzwj <https://github.com/zzzzwj>`__
* |Fix| Fix the binding problem on scheduler and optimizer when using parallelization | `@Alex-Medium <https://github.com/Alex-Medium>`__ and `@xuyxu <https://github.com/xuyxu>`__

.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |MajorFeature| replace:: :raw-html:`<span class="badge badge-success">Major Feature</span>` :raw-latex:`{\small\sc [Major Feature]}`
.. |Feature| replace:: :raw-html:`<span class="badge badge-success">Feature</span>` :raw-latex:`{\small\sc [Feature]}`
.. |Efficiency| replace:: :raw-html:`<span class="badge badge-info">Efficiency</span>` :raw-latex:`{\small\sc [Efficiency]}`
.. |Enhancement| replace:: :raw-html:`<span class="badge badge-info">Enhancement</span>` :raw-latex:`{\small\sc [Enhancement]}`
.. |Fix| replace:: :raw-html:`<span class="badge badge-danger">Fix</span>` :raw-latex:`{\small\sc [Fix]}`
.. |API| replace:: :raw-html:`<span class="badge badge-warning">API Change</span>` :raw-latex:`{\small\sc [API Change]}`
