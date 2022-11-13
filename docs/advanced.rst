Advanced Usage
==============

The following sections outline advanced usage in :mod:`torchensemble`.

Faster inference using functorch
--------------------------------

:mod:`functorch` has been integrated into Pytorch since the release of version 1.13, which is JAX-like composable function transforms for PyTorch. To enable faster inference of ensembles in :mod:`torchensemble`, you could use :meth:`vectorize` method of the ensemble to convert it into a stateless version (fmodel), and stacked parameters and buffers.

The stateless model, parameters, along with buffers could be used to reduce the inference time using :meth:`vmap` in :mod:`functorch`. More details are available at `functorch documentation <https://pytorch.org/functorch/stable/notebooks/ensembling.html>`__. The code snippet below demonstrates how to pass :meth:`ensemble.vectorize` results into :meth:`functorch.vmap`.

.. code:: python

  from torchensemble import VotingClassifier  # voting is a classic ensemble strategy