Roadmap
=======

Anyone is welcomed to work on any feature or enhancement listed below. Your contributions would be clearly posted in `Contributors <./contributors.html>`__ and `Changelog <./changelog.html>`__.

Long-term Objective
-------------------

* |MajorFeature| |API| Verify the effectiveness of new ensembles that could be included in torchensemble
* |Efficiency| Improve the training and evaluating efficiency of existing ensembles

New Functionality
-----------------

* |MajorFeature| |API| Add the partial-fit mode for all ensembles
* |MajorFeature| |API| Support manually-specified evaluation metrics when calling :meth:`predict`
* |MajorFeature| |API| Integration with :mod:`torch.utils.tensorboard` for better visualization
* |MajorFeature| |API| Support arbitrary training criteria for existing ensembles

Maintenance Related
-------------------

* |Enhancement| Refactor existing code to reduce the redundancy

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