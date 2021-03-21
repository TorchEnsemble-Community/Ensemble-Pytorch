Experiments
===========

.. warning::

  This page is still under construction. We are sorry for the delay, which is caused by the prohibitively long time on running all experiments listed below.

Experiments here were designed to evaluate the performance of each ensemble available in torchensemble. We have collected four different configurations on dataset and base estimator, as listed in the table below.

+------------------+----------------+-----------+-------------------+
|       Name       |   Estimator    |  Dataset  |    n_estimators   |
+==================+================+===========+===================+
|    LeNet@MNIST   |     LeNet-5    |   MNIST   |   5, 10, 15, 20   |
+------------------+----------------+-----------+-------------------+
|  LeNet@CIFAR-10  |     LeNet-5    |  CIFAR-10 |   5, 10, 15, 20   |
+------------------+----------------+-----------+-------------------+
|  ResNet@CIFAR-10 |    ResNet-18   |  CIFAR-10 |    2, 5, 7, 10    |
+------------------+----------------+-----------+-------------------+
| ResNet@CIFAR-100 |    ResNet-18   | CIFAR-100 |    2, 5, 7, 10    |
+------------------+----------------+-----------+-------------------+

* Data augmentations were adopted on **CIFAR-10** and **CIFAR-100** datasets.
* For **LeNet-5**, the ``Adam`` optimizer with learning rate ``1e-3`` and weight decay ``5e-4`` was used.
* For **ResNet-18**, the ``SGD`` optimizer with learning rate ``1e-1`` and weight decay ``5e-4``, along with the cosine annealing scheduler on learning rate were used.

LeNet\@MNIST
~~~~~~~~~~~~

.. image:: ./_images/lenet_mnist.png
   :align: center
   :width: 400

* MNIST is a very simple dataset, and the testing acc of a single LeNet-5 estimator is over 99
* voting and bagging are the most effective ensemble in this experiment
* bagging is even better than voting since data sampling introduces more diversity into the ensemble

LeNet\@CIFAR-10
~~~~~~~~~~~~~~~

.. image:: ./_images/lenet_cifar10.png
   :align: center
   :width: 400

* CIFAR-10 is a hard dataset for LeNet-5, and the testing acc of a single LeNet-5 estimator is around 70
* gradient boosting is the most effective ensemble, since it is a bias-reduction ensemble method
* bagging is worse than voting since less training data are available
* snapshot ensemble does not adapt well with LeNet-5 (more training epochs are needed)

ResNet\@CIFAR-10
~~~~~~~~~~~~~~~~

.. image:: ./_images/resnet_cifar10.png
   :align: center
   :width: 400

* CIFAR-10 is a relatively simple dataset for ResNet-18, and the testing acc of a single ResNet-18 estimator is around 95
* voting and snapshot ensemble are the most effective ensemble in this experiment
* snapshot ensemble is even better when taking training cost into consideration

ResNet\@CIFAR-100
~~~~~~~~~~~~~~~~~

TBA.