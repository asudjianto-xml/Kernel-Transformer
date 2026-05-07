"""Post-hoc decomposition diagnostic -- see Chapter 13.

Public API:

    from tabkernels.transparency import (
        decompose_attention,
        kernel_energy_split,
        recovery_cosine,
        post_hoc_eval,
        inspect_attention,
        AttentionReport,
    )

The five primitives compose the building blocks in
:mod:`tabkernels.core.decomposition` into the inspection workflow described
in ``affinity/book/chapters/CH13_transparency.tex``.
"""
from tabkernels.transparency.decompose import (
    AttentionReport,
    decompose_attention,
    inspect_attention,
    kernel_energy_split,
    post_hoc_eval,
    recovery_cosine,
)

__all__ = [
    "AttentionReport",
    "decompose_attention",
    "inspect_attention",
    "kernel_energy_split",
    "post_hoc_eval",
    "recovery_cosine",
]
