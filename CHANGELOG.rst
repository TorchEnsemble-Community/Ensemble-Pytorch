Changelog
=========

[Beta]
------

* |MajorFeature| Add :meth:`set_scheduler` for all ensembles | @xuyxu
* |MajorFeature| Add :class:`AdversarialTrainingClassifier` and :class:`AdversarialTrainingRegressor` | @xuyxu
* |MajorFeature| Add :class:`SnapshotEnsembleClassifier` and :class:`SnapshotEnsembleRegressor` | @xuyxu
* |MajorFeature| Add model validation and serialization | @ghost-ronin and @xuyxu
* |Enhancement| Add CI and maintenance tools | @xuyxu
* |Enhancement| Add the code coverage on codecov | @xuyxu
* |Enhancement| Add the version numbers to requirements.txt | @zackhardtoname and @xuyxu
* |Enhancement| Refactor the codes on operating tensors into an independent module | @zzzzwj
* |Enhancement| Refactor the set_optimizer into an independent method | @xuyxu
* |Enhancement| Improve the logging module | @zzzzwj
* |API| Remove the input argument ``output_dim`` from all methods | @xuyxu
* |Fix| Fix the bug in logging module when using multi-processing | @zzzzwj


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
