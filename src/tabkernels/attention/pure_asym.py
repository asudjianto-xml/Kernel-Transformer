"""Pure-Asym attention with skew-symmetric B (Chapter 3 §3.4).

Score: S_ij = x_i^T B x_j / sqrt(d_in) with B = (M - M^T) / 2 for an
unconstrained learnable M. The resulting B is skew-symmetric (B^T = -B);
B_S = 0 by construction.

Used as a deliberately-impoverished baseline to test whether skew-symmetric
kernel structure has any standalone predictive content.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import AttentionBlock
from tabkernels.core.decomposition import decompose


class PureAsymAttention(AttentionBlock):
    """Pure-skew-symmetric attention; B = (M - M^T) / 2."""

    def __init__(self, d_in: int):
        super().__init__()
        self.d_in = d_in
        self.M = nn.Parameter(torch.randn(d_in, d_in) * 0.1)

    def _B(self) -> torch.Tensor:
        return (self.M - self.M.T) / 2

    def scores(self, X_q: torch.Tensor, X_t: torch.Tensor) -> torch.Tensor:
        return X_q @ self._B() @ X_t.T / math.sqrt(self.d_in)

    def forward(self, X_q: torch.Tensor, X_t: torch.Tensor,
                y_t: torch.Tensor) -> torch.Tensor:
        S = self.scores(X_q, X_t)
        W = F.softmax(S, dim=-1)
        return W @ y_t

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return decompose(self._B())
