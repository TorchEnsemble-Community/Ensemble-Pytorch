Changelog
=========

[Beta]
------

Added
~~~~~
* [ENH] Add set_scheduler | @xuyxu
* [ENH] Add AdversarialTrainingClassifier and AdversarialTrainingRegressor | @xuyxu
* [ENH] Add SnapshotEnsembleClassifier and SnapshotEnsembleRegressor | @zzzzwj and @xuyxu
* [ENH] Add model validation and serialization | @ghost-ronin and @xuyxu
* [MNT] Add CI and maintenance tools | @xuyxu
* [MNT] Add the code coverage on codecov | @xuyxu
* [MNT] Add the version numbers to requirements.txt | @zackhardtoname and @xuyxu

Changed
~~~~~~~
* [ENH] Refactor the codes on operating tensors into an independent module | @zzzzwj
* [ENH] Refactor the set_optimizer into an independent method | @xuyxu
* [ENH] Improve the logging module | @zzzzwj

Removed
~~~~~~~
* [ENH] Remove the input parameter `output_dim` from all methods | @xuyxu

Fixed
~~~~~
* [BUG] Fix the bug in logging module when using multi-processing | @zzzzwj
