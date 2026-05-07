Getting Started
===============

Installation
------------

The package targets Python 3.11+ and depends on PyTorch, NumPy, SciPy,
scikit-learn, pandas and Matplotlib. Install from the source checkout:

.. code-block:: bash

   pip install -e .

For the documentation extras (Sphinx + theme):

.. code-block:: bash

   pip install -e .[docs]

Quick demo
----------

A 5-line tour of the public API:

.. code-block:: python

   import torch
   from tabkernels.kernels.local import GaussianKernel

   X = torch.randn(8, 4)
   K = GaussianKernel(bandwidth=1.0)(X, X)   # (8, 8) Gram matrix
   print(K.shape)

Where to look next
------------------

* :doc:`api/core` — abstract base classes locking the API contract.
* :doc:`api/kernels` — concrete kernel implementations.
* :doc:`api/architectures` — didactic reimplementations of FT-Transformer,
  SAINT, TabPFN, and TabICL used throughout Part IV of the monograph.
* :doc:`api/audits` — the audit suite that produces the experimental
  figures in Part III.
