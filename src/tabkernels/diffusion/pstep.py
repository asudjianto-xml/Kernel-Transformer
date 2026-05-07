"""$p$-step random-walk kernel (Chapter 7 §7.6).

K = P_sym^p where P_sym is the symmetric random-walk transition matrix.
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Kernel
from tabkernels.diffusion._laplacian import transition_sym


class PStepKernel(Kernel):
    """K = P_sym^p."""

    def __init__(self, p: int = 3):
        super().__init__()
        if p < 1:
            raise ValueError("p must be >= 1")
        self.p = p

    def forward(self, W: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        P = transition_sym(W)
        K = P
        for _ in range(self.p - 1):
            K = K @ P
        return K
