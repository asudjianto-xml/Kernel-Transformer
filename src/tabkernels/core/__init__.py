"""Core base classes, decomposition primitives, and metrics."""
from tabkernels.core import base, decomposition, metrics, types
from tabkernels.core.base import (
    Architecture,
    AttentionBlock,
    Audit,
    Kernel,
    Predictor,
    Prior,
    Sparsifier,
    Symmetrizer,
)
from tabkernels.core.decomposition import (
    cosine_similarity_F,
    decompose,
    energy_split,
    project_score,
)
from tabkernels.core.metrics import accuracy, label_swap_delta, mse, r_squared

__all__ = [
    "base",
    "decomposition",
    "metrics",
    "types",
    "Kernel",
    "Sparsifier",
    "Symmetrizer",
    "Predictor",
    "AttentionBlock",
    "Architecture",
    "Prior",
    "Audit",
    "decompose",
    "energy_split",
    "cosine_similarity_F",
    "project_score",
    "mse",
    "accuracy",
    "r_squared",
    "label_swap_delta",
]
