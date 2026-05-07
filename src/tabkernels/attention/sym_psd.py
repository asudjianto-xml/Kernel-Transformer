"""Sym-PSD attention with shared W_QK projection (Chapter 3 §3.4).

Setting W_Q = W_K = W_QK gives B = W_QK^T W_QK which is symmetric and PSD.
This is the shared-QK Reformer parameterisation (Kitaev 2020). Reduces
the parameter count of standard attention by half.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import AttentionBlock
from tabkernels.core.decomposition import decompose


class SymPSDAttention(AttentionBlock):
    """Shared-W_QK attention; B = W_QK^T W_QK is symmetric PSD by construction."""

    def __init__(self, d_in: int, d_emb: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_emb = d_emb
        self.W_QK = nn.Linear(d_in, d_emb, bias=False)

    def scores(self, X_q: torch.Tensor, X_t: torch.Tensor) -> torch.Tensor:
        Q = self.W_QK(X_q)
        K = self.W_QK(X_t)
        return Q @ K.T / math.sqrt(self.d_emb)

    def forward(self, X_q: torch.Tensor, X_t: torch.Tensor,
                y_t: torch.Tensor) -> torch.Tensor:
        S = self.scores(X_q, X_t)
        W = F.softmax(S, dim=-1)
        return W @ y_t

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        B = self.W_QK.weight.T @ self.W_QK.weight  # (d_in, d_in), PSD
        return decompose(B)
