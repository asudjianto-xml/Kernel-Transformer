"""Sparsification operators (Chapter 5).

Five operators that take a (typically dense) affinity matrix W and produce
a sparse W'.  All inherit from :class:`tabkernels.core.base.Sparsifier`.
"""
from tabkernels.sparsifiers.knn import KNNSparsifier
from tabkernels.sparsifiers.soft_topk import SinkhornTopKSparsifier
from tabkernels.sparsifiers.sparsemax import (
    SparsemaxSparsifier, EntmaxSparsifier, sparsemax, entmax_alpha,
)
from tabkernels.sparsifiers.eps_ball import EpsilonBallSparsifier

__all__ = [
    "KNNSparsifier",
    "SinkhornTopKSparsifier",
    "SparsemaxSparsifier",
    "EntmaxSparsifier",
    "EpsilonBallSparsifier",
    "sparsemax",
    "entmax_alpha",
]
