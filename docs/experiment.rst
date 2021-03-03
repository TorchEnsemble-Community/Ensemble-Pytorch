Experiments
===========

.. warning::

  This page will be updated once all experiments are done. We are sorry for the delay, which is caused by the prohibitively long time on running all experiments listed below.

Performance Comparison
----------------------

Experiments here were designed to evaluate the performance of each ensemble method compared against a single estimator.

Classification
~~~~~~~~~~~~~~

We have collected four different configurations on the dataset and base estimator type for performance comparison on classification, as listed in the table below.

+------------------+----------------+-----------+-------------------+
|       Name       | Base Estimator |  Dataset  |    n_estimators   |
+==================+================+===========+===================+
|    LeNet@MNIST   |     LeNet-5    |   MNIST   | 5, 10, 15, 20, 25 |
+------------------+----------------+-----------+-------------------+
|  LeNet@CIFAR-10  |     LeNet-5    |  CIFAR-10 | 5, 10, 15, 20, 25 |
+------------------+----------------+-----------+-------------------+
|  ResNet@CIFAR-10 |    ResNet-18   |  CIFAR-10 |    2, 5, 7, 10    |
+------------------+----------------+-----------+-------------------+
| ResNet@CIFAR-100 |    ResNet-18   | CIFAR-100 |    2, 5, 7, 10    |
+------------------+----------------+-----------+-------------------+

* Data augmentations were adopted on **CIFAR-10** and **CIFAR-100** datasets.
* For **LeNet-5**, the ``Adam`` optimizer with learning rate ``1e-3`` and weight decay ``5e-4`` was used.
* For **ResNet-18**, the ``SGD`` optimizer with learning rate ``1e-1`` and weight decay ``5e-4``, along with the cosine annealing scheduler on learning rate were used.
