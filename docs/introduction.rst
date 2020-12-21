Introduction
============

This page presents introduction on various ensemble methods available in Ensemble-PyTorch.

Notations
---------

- :math:`\{\mathbf{x}_i, y_i\}`: A batch of data;
- :math:`\{h_1, h_2, \cdots, h_M\}`: A set of :math:`M` base estimators;
- :math:`\mathcal{L}(\mathbf{o}_i, y_i)`: Training loss computated upon the output :math:`\mathbf{o}_i` on :math:`\mathbf{x}_i` and the ground-truth: :math:`y_i`;

Fusion
------

Voting
------

Bagging
-------

Gradient Boosting
-----------------