"""Standard scaled-dot-product attention (Chapter 3 §3.1).

Separate W_Q, W_K projections + softmax. Implements the AttentionBlock
contract: forward(X_q, X_t, y_t) -> predictions, decompose() -> (B_S, B_A).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import AttentionBlock
from tabkernels.core.decomposition import decompose


class StdAttention(AttentionBlock):
    """Standard scaled-dot-product attention with separate W_Q, W_K.

    Score: S_ij = (W_Q x_i)^T (W_K x_j) / sqrt(d_emb).
    Aggregation: softmax. Prediction: NW-style sum sum_j W_ij y_j.

    Parameters
    ----------
    d_in : int
        Input dimension.
    d_emb : int
        Embedding dimension d_h.
    """

    def __init__(self, d_in: int, d_emb: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_emb = d_emb
        self.W_Q = nn.Linear(d_in, d_emb, bias=False)
        self.W_K = nn.Linear(d_in, d_emb, bias=False)

    def scores(self, X_q: torch.Tensor, X_t: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(X_q)
        K = self.W_K(X_t)
        return Q @ K.T / math.sqrt(self.d_emb)

    def forward(self, X_q: torch.Tensor, X_t: torch.Tensor,
                y_t: torch.Tensor) -> torch.Tensor:
        S = self.scores(X_q, X_t)
        W = F.softmax(S, dim=-1)
        return W @ y_t

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        # B = W_Q^T W_K in input space, shape (d_in, d_in).
        B = self.W_Q.weight.T @ self.W_K.weight
        return decompose(B)
