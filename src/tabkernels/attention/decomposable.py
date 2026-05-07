"""Decomposable standard attention with score-mode switch (Chapter 3 §3.4, Chapter 13).

Architecturally identical to StdAttention but supports an inference-time
``score_mode`` switch that replaces the score matrix with either:
    - 'full'  : S as trained (default)
    - 'sym'   : (S + S^T) / 2  (post-hoc symmetric component)
    - 'asym'  : (S - S^T) / 2  (post-hoc skew-symmetric component)

This is the workhorse of the post-hoc decomposition diagnostic in
Chapter 13. It is exactly StdAttention at training time; the switch
only affects inference.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import AttentionBlock
from tabkernels.core.decomposition import decompose, project_score

ScoreMode = Literal["full", "sym", "asym"]


class DecomposableStdAttention(AttentionBlock):
    """StdAttention + inference-time score-mode switch.

    During training, ``score_mode`` is ignored (the model trains as standard
    attention). At inference the score matrix is projected onto the symmetric
    or skew-symmetric subspace before softmax, depending on ``score_mode``.

    Parameters
    ----------
    d_in : int
        Input dimension.
    d_emb : int
        Embedding dimension.
    score_mode : {'full', 'sym', 'asym'}
        Mode for inference-time score projection. Set after training.
    """

    def __init__(self, d_in: int, d_emb: int = 16, score_mode: ScoreMode = "full"):
        super().__init__()
        self.d_in = d_in
        self.d_emb = d_emb
        self.W_Q = nn.Linear(d_in, d_emb, bias=False)
        self.W_K = nn.Linear(d_in, d_emb, bias=False)
        self.score_mode: ScoreMode = score_mode

    def scores(self, X_q: torch.Tensor, X_t: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(X_q)
        K = self.W_K(X_t)
        return Q @ K.T / math.sqrt(self.d_emb)

    def forward(self, X_q: torch.Tensor, X_t: torch.Tensor,
                y_t: torch.Tensor) -> torch.Tensor:
        S = self.scores(X_q, X_t)
        # The score-mode projection only makes sense when X_q == X_t (square S).
        # In that case we apply project_score; otherwise we use S as-is.
        if S.shape[0] == S.shape[1] and self.score_mode != "full":
            S = project_score(S, self.score_mode)
        W = F.softmax(S, dim=-1)
        return W @ y_t

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        B = self.W_Q.weight.T @ self.W_K.weight
        return decompose(B)
