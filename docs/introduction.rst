Introduction
============

This page introduces ensemble methods available in Ensemble-PyTorch in a nutshell.

To begin with, below are some notations that will be used throughout the introduction.

- :math:`\mathcal{B} = \{\mathbf{x}_i, y_i\}_{i=1}^B`: A batch of data with :math:`B` samples;
- :math:`\{h^1, h^2, \cdots, h^m, \cdots, h^M\}`: A set of :math:`M` base estimators;
- :math:`\mathbf{o}_i^m`: The output of the base estimator :math:`h^m` on sample :math:`\mathbf{x}_i`. For regression, it is a scalar or a real-valued vector; For classification, it is a class vector with its size the number of classes;
- :math:`\mathcal{L}(\mathbf{o}_i, y_i)`: Training loss computated upon the output :math:`\mathbf{o}_i` on :math:`\mathbf{x}_i` and the ground-truth :math:`y_i`. For regression, it could be the common mean squared error; For classification, it could be the cross-entropy loss for multi-class classification.

Fusion
------

The output of fusion is the averaged output of all base estimators. Formally, given a sample :math:`\mathbf{x}_i`, the output of fusion is :math:`\mathbf{o}_i = \frac{1}{M} \sum_{m=1}^M \mathbf{o}_i^m`.

During the training stage, all base estimators in fusion are jointly trained with mini-batch gradient desecent. Given the output of fusion on a data batch :math:`\mathcal{B}`, the training loss is: :math:`\frac{1}{B} \sum_{i=1}^B \mathcal{L}(\mathbf{o}_i, y_i)`. After then, learnable parameters of all base estimator can be jointly updated with the auto-differentiation system in PyTorch and gradient descent. The figure below presents the data flow of fusion:

.. image:: ./_images/fusion.png
   :align: center
   :width: 200

Voting
------

Bagging
-------

Gradient Boosting
-----------------

Gradient boosting trains all base estimators in a sequential fashion, as the learning target of a base estimator :math:`h^m` is associated with the outputs from base estimators fitted before, i.e., :math:`\{h^1, \cdots, h^{m-1}\}`.

Given the :math:`M` base estimators fitted in gradient boosting, the output of the entire ensemble on a sample is :math:`\mathbf{o}_i = \sum_{m=1}^M \epsilon \mathbf{o}_i^m`, where :math:`\epsilon` is a pre-defined scalar in the range :math:`(0, 1]`, and known as the shrinkage rate in gradient boosting.

The training routine of the m-th base estimator can be summarized as follow:

- **Find the learning target** on each sample :math:`\mathbf{r}_i^m`: Given the summation on outputs from base estimators fitted before: :math:`\mathbf{o}_i^{[:m]}=\sum_{p=1}^{m-1} \epsilon \mathbf{o}_i^p` and the ground truth :math:`y_i`, the learning target is defined as :math:`\mathbf{r}_i^m = - \frac{\partial \mathcal{L}(\mathbf{o}_i^{[:m]},\ y_i)}{\partial \mathbf{o}_i^{[:m]}}`. Therefore, the learning target is simply the negative gradient of :math:`\mathcal{L}` with respect to the outputs from base estimators fitted before :math:`\mathbf{o}_i^{[:m]}`;
- **Train the m-th base estimator via least square regresison**, that is, the traing loss for the m-th base estimator is :math:`l^m = \frac{1}{B} \sum_{i=1}^B \|\mathbf{r}_i^m - \mathbf{o}_i^m\|_2^2`. Given :math:`l^m`, the learnable parameters of :math:`h^m` can be fitted using gradient descent;
- **Update the fitted output**: :math:`\mathbf{o}_i^{[:m+1]} = \mathbf{o}_i^{[:m]} + \epsilon \mathbf{o}_i^m`, and then move to the training routine of the (m+1)-th base estimator.

For regression with the mean squared error, :math:`\mathbf{r}_i^m = \mathbf{y}_i - \mathbf{o}_i^{[:m]}`. For classification with the cross-entropy loss, :math:`\mathbf{r}_i^m = \mathbf{y}_i - \text{softmax}(\mathbf{o}_i^{[:m]})`, where :math:`\mathbf{y}_i` is the one-hot encoded vector of the class label :math:`y_i`.

The figure below presents the data flow of gradient boosting during the training and evaluating stage, respectively. Notice that the training stage runs sequentially from the left to right.

.. image:: ./_images/gradient_boosting.png
   :align: center
   :width: 500