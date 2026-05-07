"""FT-Transformer -- a clean reimplementation of Gorishniy 2021.

This is the architecture introduced in Chapter 14 of the monograph.  Unlike
the production-grade ``rtdl`` implementation, this version is stripped to the
minimum needed to (i) train on a small tabular dataset, and (ii) feed cleanly
into the Chapter 13 transparency lens
(:mod:`tabkernels.transparency`).

The pipeline is

    (X_num, X_cat) --> FeatureTokenizer --> [CLS] + tokens
                  --> N x (FTAttentionBlock + FFN)
                  --> CLS readout --> classifier head.

The ``FTAttentionBlock`` wraps the recognised
:class:`tabkernels.audits._models.StdAttentionDecomposable` so that

* :func:`tabkernels.transparency.decompose_attention` recognises the inner
  attention class and extracts ``B = W_Q^T W_K`` per head, and
* the wrapper inherits :class:`tabkernels.core.base.AttentionBlock`, satisfying
  the architecture-level contract that ``attention_blocks()`` returns
  ``AttentionBlock`` instances with a working ``decompose()``.
"""
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from tabkernels.audits._models import StdAttentionDecomposable
from tabkernels.core.base import Architecture, AttentionBlock

__all__ = [
    "FeatureTokenizer",
    "FTAttentionBlock",
    "FTTransformer",
]


# ---------------------------------------------------------------------------
# Feature tokenizer (Eq. 14.1 in the chapter).
# ---------------------------------------------------------------------------


class FeatureTokenizer(nn.Module):
    """Embed numerical and categorical features into a sequence of tokens.

    For each numerical feature ``j`` we learn a weight vector
    ``w_j in R^{d_token}`` and a bias ``b_j in R^{d_token}``; the token is
    ``t_j = x_j * w_j + b_j``.  For each categorical feature ``c`` with
    cardinality ``K_c`` we learn a lookup ``E_c in R^{K_c x d_token}`` and the
    token is the row indexed by the integer category.  The output sequence
    concatenates numerical and categorical tokens.  See \\cref{eq:14:tokenization}.

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
            # Empty placeholders that still have ``.shape``.
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in self.cat_cardinalities]
        )
        for emb in self.cat_embeddings:
            assert isinstance(emb, nn.Embedding)  # for type narrowing
            nn.init.normal_(emb.weight, std=0.02)

    @property
    def n_tokens(self) -> int:
        """Number of feature tokens (excluding CLS)."""
        return self.n_num_features + len(self.cat_cardinalities)

    def forward(
        self,
        x_num: torch.Tensor | None,
        x_cat: torch.Tensor | None,
    ) -> torch.Tensor:
        """Tokenize a batch.

        Parameters
        ----------
        x_num : Tensor or None
            Shape ``(B, n_num_features)`` or ``None`` if there are no
            numerical features.
        x_cat : Tensor or None
            Shape ``(B, n_cat_features)`` integer tensor, or ``None``.

        Returns
        -------
        Tensor of shape ``(B, n_tokens, d_token)``.
        """
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
            # (B, F_num, 1) * (F_num, d_token) -> (B, F_num, d_token)
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
                "FeatureTokenizer needs at least one numerical or categorical "
                "feature; both n_num_features and len(cat_cardinalities) are 0."
            )
        return torch.cat(tokens, dim=1)


# ---------------------------------------------------------------------------
# Attention block wrapper (satisfies the AttentionBlock contract).
# ---------------------------------------------------------------------------


class FTAttentionBlock(AttentionBlock):
    """One pre-norm encoder block with a recognised decomposable attention.

    The forward path is the standard pre-norm transformer block:
    ``y = x + drop(attn(LN(x)))``,  ``y = y + drop(ffn(LN(y)))``.

    The ``AttentionBlock.forward(X_q, X_t, y_t)`` contract is for kernel-attention
    prediction blocks; an FT-Transformer encoder block does not predict from
    ``y_t``, so :meth:`forward` accepts a single sequence tensor ``x`` and
    returns the updated sequence.  The contract method is implemented under
    the alternate name :meth:`predict` and raises ``NotImplementedError`` to
    keep the abstract base happy without inviting misuse.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
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
        """Pre-norm encoder forward.  ``x`` has shape ``(B, N, d_model)``."""
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    def predict(  # pragma: no cover - documentary placeholder
        self, X_q: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor,
    ) -> torch.Tensor:
        """Not used by the FT-Transformer pipeline.

        FT-Transformer uses a [CLS] readout followed by a classifier head,
        not a kernel-prediction head.  The kernel-attention ``predict`` API of
        :class:`AttentionBlock` is therefore not applicable to encoder blocks.
        """
        raise NotImplementedError(
            "FTAttentionBlock is an encoder block; use forward(seq) instead."
        )

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate ``(B_S, B_A)`` over heads.

        For each head ``h``, ``B^{(h)} = W_Q^{(h)\\top} W_K^{(h)}``.  We sum
        ``B_S`` and ``B_A`` over heads to produce a single ``(d_model, d_model)``
        pair, which is the natural multi-head extension of the per-head
        decomposition in \\cref{eq:13:1}.
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


class FTTransformer(Architecture):
    """FT-Transformer (Gorishniy 2021), didactic clean reimplementation.

    Defaults match the recommended FT-Transformer configuration from the
    original paper: ``d_model=192, n_heads=8, n_layers=3, dim_ff=256,
    dropout=0.1``.

    Parameters
    ----------
    n_num_features : int
        Number of numerical features.
    cat_cardinalities : list[int]
        Categorical-feature cardinalities.  ``[]`` for purely numerical data.
    d_model : int
        Hidden width.
    n_heads : int
        Multi-head count; must divide ``d_model``.
    n_layers : int
        Number of encoder layers.
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
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_ff: int = 256,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cat_cardinalities = list(cat_cardinalities or [])
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        if n_classes <= 0:
            raise ValueError(f"n_classes must be > 0, got {n_classes}")
        if n_num_features == 0 and len(cat_cardinalities) == 0:
            raise ValueError(
                "FTTransformer needs at least one feature "
                "(n_num_features + len(cat_cardinalities) > 0)."
            )

        self.tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_model,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList(
            [
                FTAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.n_classes = n_classes
        self.dropout_p = dropout

    def forward(  # type: ignore[override]
        self,
        x_num: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logits.

        Parameters
        ----------
        x_num : Tensor or None
            ``(B, n_num_features)`` -- numerical features.  May be ``None`` if
            ``n_num_features == 0``.
        x_cat : Tensor or None
            ``(B, len(cat_cardinalities))`` integer tensor.  May be ``None`` if
            there are no categorical features.

        Returns
        -------
        Tensor of shape ``(B, n_classes)``.
        """
        tokens = self.tokenizer(x_num, x_cat)
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)  # (B, 1 + n_tokens, d_model)
        for layer in self.layers:
            seq = layer(seq)
        cls_out = self.norm(seq[:, 0, :])
        return self.head(cls_out)

    def attention_blocks(self) -> list[AttentionBlock]:
        """Return the ``FTAttentionBlock`` of every encoder layer.

        Each is a recognised :class:`AttentionBlock` whose ``decompose()``
        returns ``(B_S, B_A)`` aggregated over heads, and whose internal
        :class:`StdAttentionDecomposable` is also recognised by
        :func:`tabkernels.transparency.decompose_attention` for per-head
        extraction.
        """
        # ``self.layers`` is a ``ModuleList`` populated exclusively with
        # ``FTAttentionBlock`` (which is an ``AttentionBlock``); the
        # ``cast`` narrows ``Iterator[Module]`` for mypy.
        return [cast(AttentionBlock, layer) for layer in self.layers]
