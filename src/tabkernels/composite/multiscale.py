"""Multi-scale RBF kernel (Chapter 9 §9.3)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import Kernel


class MultiScale(Kernel):
    """Convex combination of RBF kernels at multiple bandwidths.

    Parameters
    ----------
    bandwidths : sequence of float
        Initial bandwidth values (positive).
    learnable_bandwidths : bool, default True
        If True, the bandwidths are learnable parameters (parameterised in
        log space for positivity).
    """

    def __init__(self, bandwidths: Sequence[float], learnable_bandwidths: bool = True):
        super().__init__()
        if len(bandwidths) < 1:
            raise ValueError("at least one bandwidth required")
        log_sigmas = torch.tensor([float(b) for b in bandwidths]).log()
        if learnable_bandwidths:
            self.log_sigmas = nn.Parameter(log_sigmas)
        else:
            self.register_buffer("log_sigmas", log_sigmas)
        self.beta_logits = nn.Parameter(torch.zeros(len(bandwidths)))

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        sq = ((X1[:, None] - X2[None, :]) ** 2).sum(-1)
        # Build per-scale Gram matrices.
        Ws = torch.stack(
            [torch.exp(-sq / (s.exp() ** 2)) for s in self.log_sigmas]
        )  # (M, N1, N2)
        beta = F.softmax(self.beta_logits, dim=0)  # (M,) on simplex
        return (beta.view(-1, 1, 1) * Ws).sum(dim=0)
