"""TabPFN-lite -- a didactic reimplementation of the TabPFN architecture.

Faithful to the spirit of TabPFN v1 \\citep{hollmann2023tabpfn} but stripped to
the minimum needed to demonstrate the kernel-method lens of Chapter 16:

  - Tokenise context tokens (x + y embedding) and query tokens (x + query
    marker) into a single sequence.
  - Stack of pre-norm transformer encoder layers using
    :class:`StdAttentionDecomposable` (so the post-hoc decomposition diagnostic
    of Chapter 13 sees a recognised attention class).
  - Linear regression head on the query positions.

This is a learning-targeted rather than a production reimplementation. For
production use, use the official TabPFN package; this class exists so the
book can train, inspect, and decompose a TabPFN-shaped model end-to-end.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tabkernels.audits._models import StdAttentionDecomposable, _EncoderLayer
from tabkernels.core.base import Architecture, AttentionBlock


class _RegressionTokenizer(nn.Module):
    """Continuous-y tokeniser (TabPFN's classification tokeniser uses an
    embedding; we use a linear projection so we can train regression PFNs)."""

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.feat_proj = nn.Linear(d_in, d_model)
        self.y_proj = nn.Linear(1, d_model)
        self.query_marker = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        # Inputs may be 2D (N, d) or 3D (B, N, d). Normalise to 3D.
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


class TabPFNLite(Architecture):
    """Didactic TabPFN-shaped regression PFN.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_model : int
        Hidden dimension of the transformer.
    n_heads : int
        Multi-head count. Must divide ``d_model``.
    n_layers : int
        Number of encoder layers.
    dim_ff : int
        Feedforward hidden dimension.
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
            _EncoderLayer(StdAttentionDecomposable(d_model, n_heads, dropout),
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
        """Predict y at queries using context. Trainer-compatible signature.

        Accepts (n_q, d) / (n_ctx, d) / (n_ctx,) or batched (B, n_q, d) / etc.
        Returns (n_q,) for unbatched input or (B, n_q) for batched input.
        """
        unbatched = X_q.dim() == 2
        seq = self.tokenizer(X_ctx, y_ctx, X_q)
        for layer in self.layers:
            seq = layer(seq)
        n_ctx = X_ctx.shape[-2]
        out = self.norm(seq[:, n_ctx:, :])
        out = self.head(out).squeeze(-1)  # (B, n_q)
        if unbatched:
            out = out.squeeze(0)
        return out

    def attention_blocks(self) -> list[nn.Module]:
        """Return the attention modules in each encoder layer.

        These are :class:`StdAttentionDecomposable` instances, so
        :func:`tabkernels.transparency.decompose_attention` recognises them
        and yields ``(B, B_S, B_A)`` per head per block.
        """
        return [layer.attn for layer in self.layers]
