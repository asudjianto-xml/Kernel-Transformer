"""Composite kernel operators (Chapter 9).

Multi-head, multi-scale, and multi-kernel composition.
"""
from tabkernels.composite.multihead import MultiHead, ChunkMultiHead
from tabkernels.composite.multiscale import MultiScale
from tabkernels.composite.multikernel import MultiKernel

__all__ = [
    "MultiHead",
    "ChunkMultiHead",
    "MultiScale",
    "MultiKernel",
]
