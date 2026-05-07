"""Bilinear Mahalanobis kernel with separate Q/K projections (Chapter 8 §8.2)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from tabkernels.core.base import Kernel


class BilinearQKKernel(Kernel):
    """Inner-product or Gaussian-distance kernel with separate W_Q, W_K.

    In ``mode='inner'``: k(x, y) = (W_Q x)^T (W_K y) / sqrt(d_h).
    In ``mode='gaussian'``: k(x, y) = exp(-||W_Q x - W_K y||^2).

    Parameters
    ----------
    d_in : int
        Input dimension.
    d_h : int
        Projected dimension.
    mode : {'inner', 'gaussian'}
        Default 'inner' (matches standard attention scoring).
    """

    def __init__(self, d_in: int, d_h: int = 16, mode: str = "inner"):
        super().__init__()
        if mode not in ("inner", "gaussian"):
            raise ValueError(f"mode must be 'inner' or 'gaussian', got {mode!r}")
        self.d_in = d_in
        self.d_h = d_h
        self.mode = mode
        self.W_Q = nn.Linear(d_in, d_h, bias=False)
        self.W_K = nn.Linear(d_in, d_h, bias=False)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(X1); K = self.W_K(X2)
        if self.mode == "inner":
            return Q @ K.T / math.sqrt(self.d_h)
        # Gaussian: ||W_Q x_i - W_K x_j||^2.
        sq = ((Q[:, None] - K[None, :]) ** 2).sum(-1)
        return torch.exp(-sq)

    def is_symmetric(self) -> bool:
        return False

    def is_psd(self) -> bool:
        return False
