"""Kernel families catalogued in Part II of the monograph.

Currently exports Family 1 (local kernels, Chapter 4).  Sister
chapters CH05--CH09 populate sibling subpackages
(:mod:`tabkernels.sparsifiers`, :mod:`tabkernels.symmetrizers`,
:mod:`tabkernels.diffusion`, :mod:`tabkernels.asymmetric`,
:mod:`tabkernels.composite`).
"""
from tabkernels.kernels.local import (
    CauchyKernel,
    EpanechnikovKernel,
    LearnedBandwidthRBF,
    MahalanobisKernel,
    RBFKernel,
)

__all__ = [
    "CauchyKernel",
    "EpanechnikovKernel",
    "LearnedBandwidthRBF",
    "MahalanobisKernel",
    "RBFKernel",
]
