"""Dual-channel attention: parallel sym + skew-sym branches (Chapter 3 §3.4).

Two parallel parameterisations of the bilinear form:
    B_S branch: (M_S + M_S^T) / 2  (symmetric)
    B_A branch: (M_A - M_A^T) / 2  (skew-symmetric)
The block uses B = B_S + B_A.

This is an explicit instantiation of the decomposition B = B_S + B_A
as separate trainable components. Used as a control to probe whether
the explicit branches behave differently from the implicit decomposition
of standard attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import AttentionBlock


class DualAttention(AttentionBlock):
    """Dual-channel attention with explicit sym and skew-sym branches."""

    def __init__(self, d_in: int):
        super().__init__()
        self.d_in = d_in
        self.M_S = nn.Parameter(torch.randn(d_in, d_in) * 0.1)
        self.M_A = nn.Parameter(torch.randn(d_in, d_in) * 0.1)

    def B_S(self) -> torch.Tensor:
        return (self.M_S + self.M_S.T) / 2

    def B_A(self) -> torch.Tensor:
        return (self.M_A - self.M_A.T) / 2

    def _B(self) -> torch.Tensor:
        return self.B_S() + self.B_A()

    def scores(self, X_q: torch.Tensor, X_t: torch.Tensor) -> torch.Tensor:
        return X_q @ self._B() @ X_t.T / math.sqrt(self.d_in)

    def forward(self, X_q: torch.Tensor, X_t: torch.Tensor,
                y_t: torch.Tensor) -> torch.Tensor:
        S = self.scores(X_q, X_t)
        W = F.softmax(S, dim=-1)
        return W @ y_t

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The branches are already constrained, so this is a no-op:
        return self.B_S(), self.B_A()
