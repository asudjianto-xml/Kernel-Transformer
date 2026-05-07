"""Attention primitives for kernel-smoother prediction (Chapter 3).

Five attention variants and one diagnostic variant; all subclass
:class:`tabkernels.core.base.AttentionBlock` and expose ``decompose()`` and
``energy_split()`` for the Chapter 13 transparency tooling.
"""
from tabkernels.attention.std import StdAttention
from tabkernels.attention.sym_psd import SymPSDAttention
from tabkernels.attention.sym_gen import SymGenAttention
from tabkernels.attention.pure_asym import PureAsymAttention
from tabkernels.attention.dual import DualAttention
from tabkernels.attention.decomposable import DecomposableStdAttention, ScoreMode

__all__ = [
    "StdAttention",
    "SymPSDAttention",
    "SymGenAttention",
    "PureAsymAttention",
    "DualAttention",
    "DecomposableStdAttention",
    "ScoreMode",
]
