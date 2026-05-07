"""Commute-time kernel (Chapter 7 §7.8).

K = L_comb^+ (Moore-Penrose pseudo-inverse of the combinatorial Laplacian).
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Kernel
from tabkernels.diffusion._laplacian import laplacian_comb


class CommuteTimeKernel(Kernel):
    """K = L_comb^+ via eigendecomposition (skipping the zero eigenvalue)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, W: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        L = laplacian_comb(W)
        eigvals, eigvecs = torch.linalg.eigh(L)
        # Pseudo-inverse: skip eigenvalues below eps.
        inv_vals = torch.where(eigvals > self.eps, 1.0 / eigvals.clamp_min(self.eps),
                                torch.zeros_like(eigvals))
        return eigvecs @ torch.diag(inv_vals) @ eigvecs.T
