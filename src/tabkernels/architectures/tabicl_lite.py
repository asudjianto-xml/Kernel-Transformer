"""TabICL-lite -- a didactic reimplementation of the TabICL architecture.

Faithful in spirit to TabICL v1 \\citep{qu2025tabicl} and its v2 successor
\\citep{qu2026tabiclv2}, stripped to the minimum needed for the kernel-method
lens of Chapter 17.

The key architectural distinguisher relative to TabPFN-lite (Chapter 16) is
the attention mechanism. TabICL replaces softmax with a *linear* attention
that uses a positive feature map (elu+1) -- the "scalable softmax" the v2 paper
emphasises and the same construction studied as linear attention by
\\citet{vonoswald2023transformers} and \\citet{ahn2023transformers}. Linear
attention has $O(N)$ instead of $O(N^2)$ scaling at large context, which is
the source of TabICLv2's reported 5--10x speedup over TabPFN-2.5.

For the post-hoc decomposition diagnostic of Chapter 13, the parameterisation
$B = W_Q^\\top W_K$ is identical to TabPFN-lite's; only the runtime
non-linearity differs. The :class:`LinearAttentionDecomposable` class below
inherits from :class:`StdAttentionDecomposable` so :func:`decompose_attention`
recognises it without modification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.audits._models import StdAttentionDecomposable, _EncoderLayer
from tabkernels.core.base import Architecture


class LinearAttentionDecomposable(StdAttentionDecomposable):
    """Linear-attention variant of :class:`StdAttentionDecomposable`.

    Score path: ``S = phi(Q) phi(K)^T / sqrt(d_h)`` with ``phi = elu + 1``,
    followed by row-normalisation (instead of softmax) to produce the
    attention weights. The bilinear form ``B = W_Q^T W_K`` is unchanged from
    the parent, so CH13's :func:`decompose_attention` works without
    modification.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dh = self.n_heads, self.d_h
        q = self.W_Q(x).view(B, N, H, dh)
        k = self.W_K(x).view(B, N, H, dh)
        v = self.W_V(x).view(B, N, H, dh)
        q_phi = F.elu(q) + 1.0
        k_phi = F.elu(k) + 1.0
        S = torch.einsum("bnhd,bmhd->bhnm", q_phi, k_phi) / (dh ** 0.5)
        if self.score_mode == "sym":
            S = (S + S.transpose(-1, -2)) / 2
        elif self.score_mode == "asym":
            S = (S - S.transpose(-1, -2)) / 2
        # Row-stochastic normalisation (no softmax).
        W = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
        out = torch.einsum("bhnm,bmhd->bnhd", W, v).reshape(B, N, H * dh)
        return self.dropout(self.W_O(out))


class _RegressionTokenizer(nn.Module):
    """Continuous-y tokeniser identical in shape to the TabPFN-lite tokeniser."""

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.feat_proj = nn.Linear(d_in, d_model)
        self.y_proj = nn.Linear(1, d_model)
        self.query_marker = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        if X_ctx.dim() == 2:
            X_ctx = X_ctx.unsqueeze(0)
            X_q = X_q.unsqueeze(0)
            y_ctx = y_ctx.unsqueeze(0)
        if y_ctx.dim() == 2:
            y_emb_in = y_ctx.unsqueeze(-1)
        else:
            y_emb_in = y_ctx
        h_ctx = self.feat_proj(X_ctx) + self.y_proj(y_emb_in)
        h_q = self.feat_proj(X_q) + self.query_marker
        return torch.cat([h_ctx, h_q], dim=1)


class TabICLLite(Architecture):
    """Didactic TabICL-shaped regression PFN with linear attention.

    Identical wiring to :class:`TabPFNLite` (\\cref{ch:16}) except every
    encoder layer's attention block is :class:`LinearAttentionDecomposable`.
    This isolates the architectural difference -- linear vs softmax
    attention -- so the post-hoc decomposition diagnostic (Chapter 13)
    sees only the effect of the non-linearity choice.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_model : int
        Hidden dimension.
    n_heads : int
        Multi-head count. Must divide ``d_model``.
    n_layers : int
        Number of encoder layers.
    dim_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_in: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dim_ff: int = 128, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.tokenizer = _RegressionTokenizer(d_in=d_in, d_model=d_model)
        self.layers = nn.ModuleList([
            _EncoderLayer(LinearAttentionDecomposable(d_model, n_heads, dropout),
                          d_model, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

    def forward(self, X_q: torch.Tensor, X_ctx: torch.Tensor,
                y_ctx: torch.Tensor) -> torch.Tensor:
        unbatched = X_q.dim() == 2
        seq = self.tokenizer(X_ctx, y_ctx, X_q)
        for layer in self.layers:
            seq = layer(seq)
        n_ctx = X_ctx.shape[-2]
        out = self.norm(seq[:, n_ctx:, :])
        out = self.head(out).squeeze(-1)
        if unbatched:
            out = out.squeeze(0)
        return out

    def attention_blocks(self) -> list[nn.Module]:
        return [layer.attn for layer in self.layers]
