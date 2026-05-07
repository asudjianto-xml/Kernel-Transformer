"""SAINT -- a clean reimplementation of Somepalli et al. 2021.

This is the architecture introduced in Chapter 15 of the monograph.  Unlike
FT-Transformer (\\cref{ch:14}) -- a single token-axis transformer -- SAINT
applies attention along **two axes** in alternation:

* a **column** axis (intra-sample feature attention), shape ``(B, F, d)``
  where each sample's tokens attend among themselves -- this is the same
  shape as an FT-Transformer block and gives a per-sample feature kernel;
* a **row** axis (inter-sample attention), shape ``(F * d) -> (B, F * d)``
  reshaped so the *batch* dimension becomes the sequence dimension and each
  sample is a single token.

The pipeline is

    (X_num, X_cat) --> FeatureTokenizer --> tokens (B, F, d)
                  --> N x [ColumnAttentionBlock + RowAttentionBlock + FFN]
                  --> mean-pool over tokens --> classifier head.

Both attention blocks wrap the recognised
:class:`tabkernels.audits._models.StdAttentionDecomposable` so that

* :func:`tabkernels.transparency.decompose_attention` recognises the inner
  attention class and extracts ``B = W_Q^T W_K`` per head, and
* the wrappers inherit :class:`tabkernels.core.base.AttentionBlock`,
  satisfying the architecture-level contract that ``attention_blocks()``
  returns ``AttentionBlock`` instances with a working ``decompose()``.

The two block types carry an ``axis`` attribute (``"row"`` or ``"column"``)
so the Chapter 13 lens can be applied separately to each axis -- the
distinguishing contribution of this chapter.
"""
from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn as nn

from tabkernels.audits._models import StdAttentionDecomposable
from tabkernels.core.base import Architecture, AttentionBlock

__all__ = [
    "SAINTFeatureTokenizer",
    "SAINTAttentionBlock",
    "SAINT",
]


# ---------------------------------------------------------------------------
# Feature tokenizer (same shape as FT-Transformer's; SAINT reuses it).
# ---------------------------------------------------------------------------


class SAINTFeatureTokenizer(nn.Module):
    """Embed numerical and categorical features into a sequence of tokens.

    Identical to the FT-Transformer tokenizer (\\cref{eq:14:tokenization}):
    ``t_j = x_j * w_j + b_j`` for numerical, ``E_c[x_c]`` for categorical.

    Parameters
    ----------
    n_num_features : int
        Number of numerical features.  May be 0.
    cat_cardinalities : list[int]
        Cardinalities of the categorical features.  May be ``[]``.
    d_token : int
        Embedding width.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        d_token: int,
    ) -> None:
        super().__init__()
        if n_num_features < 0:
            raise ValueError(f"n_num_features must be >= 0, got {n_num_features}")
        if d_token <= 0:
            raise ValueError(f"d_token must be > 0, got {d_token}")
        if any(c <= 0 for c in cat_cardinalities):
            raise ValueError("cat_cardinalities entries must all be > 0")
        self.n_num_features = n_num_features
        self.cat_cardinalities = list(cat_cardinalities)
        self.d_token = d_token

        if n_num_features > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num_features, d_token))
            self.num_bias = nn.Parameter(torch.empty(n_num_features, d_token))
            nn.init.normal_(self.num_weight, std=0.02)
            nn.init.zeros_(self.num_bias)
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in self.cat_cardinalities]
        )
        for emb in self.cat_embeddings:
            assert isinstance(emb, nn.Embedding)
            nn.init.normal_(emb.weight, std=0.02)

    @property
    def n_tokens(self) -> int:
        return self.n_num_features + len(self.cat_cardinalities)

    def forward(
        self,
        x_num: torch.Tensor | None,
        x_cat: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        batch_size: int | None = None

        if self.n_num_features > 0:
            if x_num is None:
                raise ValueError("x_num is required (n_num_features > 0)")
            if x_num.dim() != 2 or x_num.shape[1] != self.n_num_features:
                raise ValueError(
                    f"x_num must have shape (B, {self.n_num_features}), "
                    f"got {tuple(x_num.shape)}"
                )
            batch_size = x_num.shape[0]
            num_tokens = (
                x_num.unsqueeze(-1) * self.num_weight + self.num_bias
            )
            tokens.append(num_tokens)

        if len(self.cat_cardinalities) > 0:
            if x_cat is None:
                raise ValueError("x_cat is required (len(cat_cardinalities) > 0)")
            if x_cat.dim() != 2 or x_cat.shape[1] != len(self.cat_cardinalities):
                raise ValueError(
                    f"x_cat must have shape (B, {len(self.cat_cardinalities)}), "
                    f"got {tuple(x_cat.shape)}"
                )
            batch_size = x_cat.shape[0] if batch_size is None else batch_size
            cat_tokens_list = [
                emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)
            ]
            cat_tokens = torch.stack(cat_tokens_list, dim=1)
            tokens.append(cat_tokens)

        if not tokens:
            raise ValueError(
                "SAINTFeatureTokenizer needs at least one numerical or categorical "
                "feature; both n_num_features and len(cat_cardinalities) are 0."
            )
        return torch.cat(tokens, dim=1)


# ---------------------------------------------------------------------------
# Dual-axis attention block wrapper.
# ---------------------------------------------------------------------------


class SAINTAttentionBlock(AttentionBlock):
    """One pre-norm encoder block, parameterised by an attention ``axis``.

    The forward path is the standard pre-norm transformer block:
    ``y = x + drop(attn(LN(x)))`` then ``y = y + drop(ffn(LN(y)))``.

    The ``axis`` attribute is one of ``"row"`` (inter-sample) or ``"column"``
    (intra-sample feature) and determines how the tokens are reshaped before
    the attention operation:

    * ``"column"``: input shape ``(B, F, d_model)``; attention is computed
      across the ``F`` token axis -- the same shape as an FT-Transformer
      block.
    * ``"row"``: input shape ``(B, F, d_model)`` is reshaped to
      ``(1, B, F * d_model)`` so the *batch* axis becomes the sequence
      axis; the attention runs at width ``F * d_model``.  Each sample is
      a single token of size ``F * d_model``.

    For the kernel-lens decomposition this means the ``column``-axis block
    has a ``(d_model, d_model)`` bilinear form $B$, while the ``row``-axis
    block has a ``(F * d_model, F * d_model)`` bilinear form.  We accumulate
    per-head $\\BS, \\BA$ over heads as in :class:`FTAttentionBlock`.

    Parameters
    ----------
    d_model : int
        Hidden width of the wrapped attention.  For column blocks this is
        ``d_token``; for row blocks this is ``F * d_token``.
    n_heads : int
        Multi-head count; must divide ``d_model``.
    dim_ff : int
        Feedforward hidden width.
    dropout : float
        Dropout rate.
    axis : {"row", "column"}
        Which axis the block attends along.
    """

    axis: Literal["row", "column"]

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int = 256,
        dropout: float = 0.1,
        axis: Literal["row", "column"] = "column",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        if axis not in ("row", "column"):
            raise ValueError(f"axis must be 'row' or 'column', got {axis!r}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.axis = axis
        self.attn = StdAttentionDecomposable(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, *_: object, **__: object,
    ) -> torch.Tensor:
        """Pre-norm encoder forward.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, F, d_token)``.

        Returns
        -------
        Tensor of the same shape as ``x``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"SAINTAttentionBlock expects (B, F, d_token); got {tuple(x.shape)}"
            )
        B, F, d = x.shape
        if self.axis == "column":
            # Per-sample feature attention; sequence is F tokens of width d.
            assert d == self.d_model, (
                f"column-axis block expects d_model={self.d_model}, got d_token={d}"
            )
            seq = x  # (B, F, d_model)
            seq = seq + self.dropout(self.attn(self.norm1(seq)))
            seq = seq + self.dropout(self.ffn(self.norm2(seq)))
            return seq
        # Row-axis: each sample is a single token of width F*d.
        assert F * d == self.d_model, (
            f"row-axis block expects d_model={self.d_model}, got F*d={F * d}"
        )
        seq = x.reshape(1, B, F * d)  # (1, B, F*d)
        seq = seq + self.dropout(self.attn(self.norm1(seq)))
        seq = seq + self.dropout(self.ffn(self.norm2(seq)))
        return seq.reshape(B, F, d)

    def predict(  # pragma: no cover - documentary placeholder
        self, X_q: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor,
    ) -> torch.Tensor:
        """Not used by the SAINT pipeline.

        SAINT uses a mean-pool readout followed by a classifier head, not a
        kernel-prediction head.  The kernel-attention ``predict`` API of
        :class:`AttentionBlock` is therefore not applicable to encoder blocks.
        """
        raise NotImplementedError(
            "SAINTAttentionBlock is an encoder block; use forward(seq) instead."
        )

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate ``(B_S, B_A)`` over heads.

        For each head ``h``, ``B^{(h)} = W_Q^{(h)\\top} W_K^{(h)}``.  We sum
        ``B_S`` and ``B_A`` over heads to produce a single ``(d, d)`` pair,
        with ``d = self.d_model`` -- so column blocks return
        ``(d_token, d_token)`` and row blocks return ``(F*d_token, F*d_token)``.
        """
        H, dh = self.attn.n_heads, self.attn.d_h
        d_model = self.attn.W_Q.weight.shape[0]
        WQ = self.attn.W_Q.weight.view(H, dh, d_model)
        WK = self.attn.W_K.weight.view(H, dh, d_model)
        B_S = torch.zeros(d_model, d_model, device=WQ.device, dtype=WQ.dtype)
        B_A = torch.zeros(d_model, d_model, device=WQ.device, dtype=WQ.dtype)
        for h in range(H):
            B_h = WQ[h].T @ WK[h]
            B_S = B_S + 0.5 * (B_h + B_h.T)
            B_A = B_A + 0.5 * (B_h - B_h.T)
        return B_S.detach(), B_A.detach()


# ---------------------------------------------------------------------------
# Top-level architecture.
# ---------------------------------------------------------------------------


class SAINT(Architecture):
    """SAINT (Somepalli 2021), didactic clean reimplementation.

    Each of the ``n_layers`` "stages" applies (column attention, then row
    attention) in sequence, mirroring the alternating dual-axis design of
    \\citet{somepalli2021saint}.  Each axis carries its own
    :class:`SAINTAttentionBlock`, so ``attention_blocks()`` returns a list of
    length ``2 * n_layers``.  ``row_attention_blocks()`` and
    ``column_attention_blocks()`` expose the two halves separately.

    The readout is a mean-pool over feature tokens followed by a linear
    classifier head.  This is simpler than the original SAINT readout (which
    optionally uses contrastive auxiliary heads); for the chapter's
    decomposition lens we only need the supervised forward path.

    Parameters
    ----------
    n_num_features : int
        Number of numerical features.
    cat_cardinalities : list[int]
        Categorical-feature cardinalities.  ``[]`` for purely numerical data.
    n_samples : int
        Fixed batch / sample-axis length used for row-axis attention.  The
        row-axis block expects exactly this many samples per forward call;
        inputs of a different size are caller-padded or batched on the
        outside.  This is the design choice that distinguishes SAINT from
        FT-Transformer: ``n_samples`` parameterises the row-axis attention
        width.
    d_token : int
        Hidden width per token.
    n_heads : int
        Multi-head count; must divide ``d_token`` and ``n_samples * d_token``.
    n_layers : int
        Number of (column, row) attention stages.
    dim_ff : int
        Feedforward hidden width.
    n_classes : int
        Output classes; pass ``1`` for regression.
    dropout : float
        Dropout rate, applied inside attention, FFN, and residual paths.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int] | None = None,
        n_samples: int = 64,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 64,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cat_cardinalities = list(cat_cardinalities or [])
        if d_token % n_heads != 0:
            raise ValueError(
                f"d_token={d_token} must be divisible by n_heads={n_heads}"
            )
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        if n_classes <= 0:
            raise ValueError(f"n_classes must be > 0, got {n_classes}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")
        if n_num_features == 0 and len(cat_cardinalities) == 0:
            raise ValueError(
                "SAINT needs at least one feature "
                "(n_num_features + len(cat_cardinalities) > 0)."
            )
        F_total = n_num_features + len(cat_cardinalities)
        d_row = F_total * d_token
        if d_row % n_heads != 0:
            raise ValueError(
                f"row-axis width F*d_token={d_row} must be divisible by "
                f"n_heads={n_heads}"
            )

        self.tokenizer = SAINTFeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
        )

        self.column_layers = nn.ModuleList(
            [
                SAINTAttentionBlock(
                    d_model=d_token,
                    n_heads=n_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                    axis="column",
                )
                for _ in range(n_layers)
            ]
        )
        self.row_layers = nn.ModuleList(
            [
                SAINTAttentionBlock(
                    d_model=d_row,
                    n_heads=n_heads,
                    dim_ff=max(dim_ff, d_row),
                    dropout=dropout,
                    axis="row",
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.n_samples = n_samples
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.n_classes = n_classes
        self.dropout_p = dropout
        self.F_total = F_total

    def forward(
        self,
        x_num: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logits.

        Parameters
        ----------
        x_num : Tensor or None
            ``(B, n_num_features)`` numerical features.  May be ``None`` if
            ``n_num_features == 0``.
        x_cat : Tensor or None
            ``(B, len(cat_cardinalities))`` integer tensor.  May be ``None``
            if there are no categorical features.

        Returns
        -------
        Tensor of shape ``(B, n_classes)``.
        """
        seq = self.tokenizer(x_num, x_cat)  # (B, F, d_token)
        for col_layer, row_layer in zip(
            self.column_layers, self.row_layers, strict=True
        ):
            seq = col_layer(seq)
            seq = row_layer(seq)
        # Mean-pool over feature tokens, then linear head.
        pooled = self.norm(seq.mean(dim=1))  # (B, d_token)
        return self.head(pooled)

    def attention_blocks(self) -> list[AttentionBlock]:
        """Return *all* attention blocks (column + row), in execution order.

        For each of the ``n_layers`` stages, the column block comes first
        and the row block second; the returned list has length ``2 *
        n_layers``.  Each block carries an ``axis`` attribute.
        """
        out: list[AttentionBlock] = []
        for col_layer, row_layer in zip(
            self.column_layers, self.row_layers, strict=True
        ):
            out.append(cast(AttentionBlock, col_layer))
            out.append(cast(AttentionBlock, row_layer))
        return out

    def column_attention_blocks(self) -> list[AttentionBlock]:
        """Return only the column-axis (intra-sample feature) blocks."""
        return [cast(AttentionBlock, layer) for layer in self.column_layers]

    def row_attention_blocks(self) -> list[AttentionBlock]:
        """Return only the row-axis (inter-sample) blocks."""
        return [cast(AttentionBlock, layer) for layer in self.row_layers]
