"""Regularised Laplacian kernel (Chapter 7 §7.4)."""
from __future__ import annotations

import torch

from tabkernels.core.base import Kernel
from tabkernels.diffusion._laplacian import laplacian_sym


class RegLaplacianKernel(Kernel):
    """K = (I + alpha L_sym)^{-1}."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = alpha

    def forward(self, W: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        L = laplacian_sym(W)
        N = W.shape[0]
        I = torch.eye(N, device=W.device, dtype=W.dtype)
        return torch.linalg.solve(I + self.alpha * L, I)
