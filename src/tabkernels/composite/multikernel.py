"""Multi-kernel composition (Chapter 9 §9.4)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import Kernel


class MultiKernel(Kernel):
    """Convex combination of arbitrary base kernels (multiple kernel learning).

    Parameters
    ----------
    kernels : sequence of Kernel
        Base kernels. Each must accept the same input shape.
    learnable_weights : bool, default True
        If True, mixing weights are learnable (parameterised via softmax to
        enforce convex combination).
    """

    def __init__(self, kernels: Sequence[Kernel], learnable_weights: bool = True):
        super().__init__()
        if len(kernels) < 1:
            raise ValueError("at least one kernel required")
        self.kernels = nn.ModuleList(kernels)
        self.beta_logits = nn.Parameter(torch.zeros(len(kernels)))
        if not learnable_weights:
            self.beta_logits.requires_grad_(False)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        Ws = torch.stack([k(X1, X2) for k in self.kernels])  # (M, N1, N2)
        beta = F.softmax(self.beta_logits, dim=0)
        return (beta.view(-1, 1, 1) * Ws).sum(dim=0)

    def weights(self) -> torch.Tensor:
        """Return the current convex-combination weights (after softmax)."""
        return F.softmax(self.beta_logits, dim=0).detach()
