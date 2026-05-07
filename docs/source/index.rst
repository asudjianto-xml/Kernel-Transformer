tabkernels
==========

``tabkernels`` is the companion Python package to the monograph
*Kernels and Transformers for Tabular Data*. It collects the kernels,
sparsifiers, symmetrizers, attention blocks, audits, priors, and didactic
architecture reimplementations (FT-Transformer, SAINT, TabPFN, TabICL) used
throughout the book, behind a small set of abstract base classes
(:class:`~tabkernels.core.base.Kernel`,
:class:`~tabkernels.core.base.Sparsifier`,
:class:`~tabkernels.core.base.Symmetrizer`, ...).

The package is intentionally pedagogical: implementations favour clarity over
speed, and every module corresponds to a chapter or section of the book.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
