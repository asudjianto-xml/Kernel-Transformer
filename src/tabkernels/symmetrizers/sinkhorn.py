"""Sinkhorn / bistochastic symmetrisation (Chapter 6 §6.5).

By the Sinkhorn--Knopp theorem (1967), iteratively row- and column-normalising
a positive matrix converges to a unique doubly-stochastic matrix.
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Symmetrizer


class SinkhornSymmetrizer(Symmetrizer):
    """Iterative bistochastic normalisation (Eq. 6.4).

    Parameters
    ----------
    iters : int, default 20
        Number of Sinkhorn iterations.
    eps : float, default 1e-8
        Floor on row/column sums to avoid division by zero.
    """

    def __init__(self, iters: int = 20, eps: float = 1e-8):
        super().__init__()
        if iters < 1:
            raise ValueError("iters must be >= 1")
        self.iters = iters
        self.eps = eps

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square, got {tuple(W.shape)}")
        for _ in range(self.iters):
            W = W / W.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            W = W / W.sum(dim=0, keepdim=True).clamp_min(self.eps)
        # Final symmetrisation pass to ensure W = W^T (the iteration converges
        # to a bistochastic matrix; for symmetric inputs this is symmetric, but
        # for asymmetric inputs we additionally apply additive symmetrisation).
        return 0.5 * (W + W.transpose(-2, -1))
