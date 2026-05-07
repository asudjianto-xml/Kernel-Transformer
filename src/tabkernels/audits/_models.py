"""Private model classes used by the migrated audits.

PLACEHOLDER STATUS
==================
These classes are placeholders for what will eventually live under
``tabkernels.attention.*`` once AGENT-CH03 lands. They are deliberately kept
private (leading underscore module name) so chapter agents do not accidentally
import them as a public API.

Once CH03 publishes the proper :class:`tabkernels.core.base.AttentionBlock`
subclasses (Std, SymPSD, SymGen, PureAsym, Dual, plus the ICL transformer
wrappers), the audits in this package should switch to importing from there
and this module should be deleted.

The classes here mirror what was hand-written across the original
``~/jupyterlab/ICL/*.py`` scripts; they are deterministic given a seed and have
been kept faithful to the original implementations so the cached JSON results
remain reproducible bit-for-bit (within seed noise).

CH03 punch list (classes that need a public home in ``tabkernels.attention``):

* :class:`KernelAttention`            -- the 5-variant ``(std, sym_psd, sym_gen,
                                         pure_asym, dual) x (softmax,
                                         nw_performer)`` block used by Oracle and
                                         Directional audits.
* :class:`PairwisePredictor`          -- the ``z^T A z`` pairwise scorer used by
                                         the Pairwise audits.
* :class:`StdAttentionDecomposable`   -- standard separate-W_Q/W_K softmax
                                         attention with a post-hoc score-mode
                                         switch (used by PostHocDecomposition).
* :class:`PureAsymAttention`          -- skew-symmetric scoring head used by the
                                         ICL audits.
* The ``*ICLTransformer`` wrappers and their encoder layers.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Tabular kernel-attention block (used by OracleAudit, DirectionalAudit,
# StructureAnalysis).
# ----------------------------------------------------------------------------


class KernelAttention(nn.Module):
    """Single-block kernel attention with a 5-variant kernel parameterisation.

    The kernel score is one of:
      * ``std``       -- ``W_Q x . W_K y / sqrt(d_emb)``
      * ``sym_psd``   -- ``W_QK x . W_QK y / sqrt(d_emb)`` (PSD by construction)
      * ``sym_gen``   -- ``x^T (M+M^T)/2 y / sqrt(d_in)``
      * ``pure_asym`` -- ``x^T (M-M^T)/2 y / sqrt(d_in)``
      * ``dual``      -- ``x^T (B_S + B_A) y / sqrt(d_in)``

    Aggregation is one of ``softmax`` or ``nw_performer`` (Performer-style
    positive feature maps with row normalisation).
    """

    def __init__(self, d_in: int, d_emb: int = 16, variant: str = "std",
                 aggregation: str = "softmax") -> None:
        super().__init__()
        self.variant = variant
        self.aggregation = aggregation
        self.d_emb = d_emb
        self.d_in = d_in
        if variant == "std":
            self.W_Q = nn.Linear(d_in, d_emb, bias=False)
            self.W_K = nn.Linear(d_in, d_emb, bias=False)
        elif variant == "sym_psd":
            self.W_QK = nn.Linear(d_in, d_emb, bias=False)
        elif variant in ("sym_gen", "pure_asym"):
            self.M = nn.Parameter(torch.randn(d_in, d_in) * 0.1)
        elif variant == "dual":
            self.M_S = nn.Parameter(torch.randn(d_in, d_in) * 0.1)
            self.M_A = nn.Parameter(torch.randn(d_in, d_in) * 0.1)
        else:
            raise ValueError(variant)
        self.log_tau = nn.Parameter(torch.zeros(1))

    def kernel_scores(self, X_q: torch.Tensor, X_t: torch.Tensor) -> torch.Tensor:
        if self.variant == "std":
            S = self.W_Q(X_q) @ self.W_K(X_t).T / math.sqrt(self.d_emb)
        elif self.variant == "sym_psd":
            Z_q = self.W_QK(X_q)
            Z_t = self.W_QK(X_t)
            S = Z_q @ Z_t.T / math.sqrt(self.d_emb)
        elif self.variant == "sym_gen":
            B = (self.M + self.M.T) / 2
            S = X_q @ B @ X_t.T / math.sqrt(self.d_in)
        elif self.variant == "pure_asym":
            B = (self.M - self.M.T) / 2
            S = X_q @ B @ X_t.T / math.sqrt(self.d_in)
        elif self.variant == "dual":
            B_S = (self.M_S + self.M_S.T) / 2
            B_A = (self.M_A - self.M_A.T) / 2
            S = X_q @ (B_S + B_A) @ X_t.T / math.sqrt(self.d_in)
        else:  # pragma: no cover -- defensive
            raise ValueError(self.variant)
        return S * torch.exp(self.log_tau)

    def attention_weights(self, X_q: torch.Tensor, X_t: torch.Tensor,
                          mask_diag: bool = False,
                          order_q: torch.Tensor | None = None,
                          order_t: torch.Tensor | None = None,
                          causal: bool = False) -> torch.Tensor:
        N_q = X_q.shape[0]
        N_t = X_t.shape[0]
        if self.aggregation == "softmax":
            S = self.kernel_scores(X_q, X_t)
            if mask_diag and N_q == N_t:
                S = S - torch.eye(N_q, device=S.device) * 1e9
            if causal and order_q is not None and order_t is not None:
                mask_no = order_q.unsqueeze(1) <= order_t.unsqueeze(0)
                S = S.masked_fill(mask_no, -1e9)
            W = torch.softmax(S, dim=-1)
            W = torch.nan_to_num(W, nan=0.0)
            return W
        if self.aggregation == "nw_performer":
            if self.variant == "std":
                phi_q = F.elu(self.W_Q(X_q)) + 1
                phi_t = F.elu(self.W_K(X_t)) + 1
            elif self.variant == "sym_psd":
                phi_q = F.elu(self.W_QK(X_q)) + 1
                phi_t = F.elu(self.W_QK(X_t)) + 1
            else:
                raise NotImplementedError(
                    f"nw_performer not implemented for {self.variant}")
            S = phi_q @ phi_t.T
            if mask_diag and N_q == N_t:
                S = S * (1 - torch.eye(N_q, device=S.device))
            S = S / (S.sum(-1, keepdim=True) + 1e-8)
            return S
        raise ValueError(self.aggregation)

    def predict(self, X_q: torch.Tensor, X_t: torch.Tensor, y_t: torch.Tensor,
                mask_diag: bool = False,
                order_q: torch.Tensor | None = None,
                order_t: torch.Tensor | None = None,
                causal: bool = False) -> torch.Tensor:
        W = self.attention_weights(
            X_q, X_t, mask_diag=mask_diag,
            order_q=order_q, order_t=order_t, causal=causal,
        )
        return W @ y_t


# ----------------------------------------------------------------------------
# Pairwise predictor (used by PairwiseAudit).
# ----------------------------------------------------------------------------


class PairwisePredictor(nn.Module):
    """``R_{ij} = z_i^T A z_j`` with five constraint variants on ``A``.

    Variants: ``sym_psd`` (``W^T W``), ``sym_gen`` (``(M+M^T)/2``),
    ``pure_asym`` (``(M-M^T)/2``), ``full`` (``M``), ``dual`` (sum of
    independently-parameterised sym + skew).
    """

    def __init__(self, d_in: int, variant: str = "full") -> None:
        super().__init__()
        self.variant = variant
        if variant == "sym_psd":
            self.W = nn.Parameter(torch.randn(d_in, d_in) * 0.1)
        elif variant in ("sym_gen", "pure_asym", "full"):
            self.M = nn.Parameter(torch.randn(d_in, d_in) * 0.1)
        elif variant == "dual":
            self.M_S = nn.Parameter(torch.randn(d_in, d_in) * 0.1)
            self.M_A = nn.Parameter(torch.randn(d_in, d_in) * 0.1)
        else:
            raise ValueError(variant)

    def kernel_matrix(self) -> torch.Tensor:
        if self.variant == "sym_psd":
            return self.W.T @ self.W
        if self.variant == "sym_gen":
            return (self.M + self.M.T) / 2
        if self.variant == "pure_asym":
            return (self.M - self.M.T) / 2
        if self.variant == "full":
            return self.M
        if self.variant == "dual":
            return ((self.M_S + self.M_S.T) / 2
                    + (self.M_A - self.M_A.T) / 2)
        raise ValueError(self.variant)  # pragma: no cover

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return Z @ self.kernel_matrix() @ Z.T


# ----------------------------------------------------------------------------
# ICL transformer building blocks (used by ThreeWayAblation, StandalonePureAsym,
# ResidualBoost, DualChannelAudit, PostHocDecomposition).
# ----------------------------------------------------------------------------


class TabularTokenizer(nn.Module):
    """Embed ``(X_ctx, y_ctx)`` and ``X_q`` into a shared sequence."""

    def __init__(self, d_x: int, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(d_x, d_model)
        self.label_emb = nn.Embedding(n_classes, d_model)
        self.query_emb = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        h_ctx = self.feat_proj(X_ctx) + self.label_emb(y_ctx)
        h_q = self.feat_proj(X_q) + self.query_emb
        return torch.cat([h_ctx, h_q], dim=1)


class StdICLTransformer(nn.Module):
    """Standard PyTorch encoder block with separate W_Q, W_K (asymmetric softmax)."""

    def __init__(self, d_x: int = 8, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 1, dim_ff: int = 128, dropout: float = 0.0,
                 n_classes: int = 2) -> None:
        super().__init__()
        self.tokenizer = TabularTokenizer(d_x, d_model, n_classes)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)
        self.n_classes = n_classes

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        seq = self.tokenizer(X_ctx, y_ctx, X_q)
        out = self.encoder(seq)
        N_c = X_ctx.size(1)
        return self.head(out[:, N_c:, :])


class PSDAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.W_QK = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dh = self.n_heads, self.d_h
        z = self.W_QK(x).view(B, N, H, dh)
        v = self.W_V(x).view(B, N, H, dh)
        phi = F.elu(z) + 1.0
        W = torch.einsum("bnhd,bmhd->bhnm", phi, phi) / dh
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
        out = torch.einsum("bhnm,bmhd->bnhd", W, v).reshape(B, N, H * dh)
        return self.dropout(self.W_O(out))


class _EncoderLayer(nn.Module):
    """Generic ``norm -> attn -> add -> norm -> ffn -> add`` block."""

    def __init__(self, attn: nn.Module, d_model: int, dim_ff: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PSDICLTransformer(nn.Module):
    def __init__(self, d_x: int = 8, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 1, dim_ff: int = 128, dropout: float = 0.0,
                 n_classes: int = 2) -> None:
        super().__init__()
        self.tokenizer = TabularTokenizer(d_x, d_model, n_classes)
        self.layers = nn.ModuleList([
            _EncoderLayer(PSDAttention(d_model, n_heads, dropout),
                          d_model, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)
        self.n_classes = n_classes

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        seq = self.tokenizer(X_ctx, y_ctx, X_q)
        for layer in self.layers:
            seq = layer(seq)
        N_c = X_ctx.size(1)
        return self.head(seq[:, N_c:, :])


class SymSoftmaxAttention(nn.Module):
    """Shared ``W_QK`` produces symmetric softmax attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.W_QK = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dh = self.n_heads, self.d_h
        z = self.W_QK(x).view(B, N, H, dh)
        v = self.W_V(x).view(B, N, H, dh)
        scores = torch.einsum("bnhd,bmhd->bhnm", z, z) / (dh ** 0.5)
        W = F.softmax(scores, dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", W, v).reshape(B, N, H * dh)
        return self.dropout(self.W_O(out))


class SymSoftmaxICLTransformer(nn.Module):
    def __init__(self, d_x: int = 8, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 1, dim_ff: int = 128, dropout: float = 0.0,
                 n_classes: int = 2) -> None:
        super().__init__()
        self.tokenizer = TabularTokenizer(d_x, d_model, n_classes)
        self.layers = nn.ModuleList([
            _EncoderLayer(SymSoftmaxAttention(d_model, n_heads, dropout),
                          d_model, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)
        self.n_classes = n_classes

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        seq = self.tokenizer(X_ctx, y_ctx, X_q)
        for layer in self.layers:
            seq = layer(seq)
        N_c = X_ctx.size(1)
        return self.head(seq[:, N_c:, :])


class PureAsymAttention(nn.Module):
    """Skew-symmetric scoring head: ``S = z^T (M - M^T) z``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.W_E = nn.Linear(d_model, d_model, bias=False)
        self.M = nn.Parameter(torch.randn(n_heads, self.d_h, self.d_h) * 0.02)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dh = self.n_heads, self.d_h
        z = self.W_E(x).view(B, N, H, dh)
        v = self.W_V(x).view(B, N, H, dh)
        A = self.M - self.M.transpose(-1, -2)
        scores = torch.einsum("bnhd,hde,bmhe->bhnm", z, A, z) / (dh ** 0.5)
        W = F.softmax(scores, dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", W, v).reshape(B, N, H * dh)
        return self.dropout(self.W_O(out))


class PureAsymICLTransformer(nn.Module):
    def __init__(self, d_x: int = 8, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 1, dim_ff: int = 128, dropout: float = 0.0,
                 n_classes: int = 2) -> None:
        super().__init__()
        self.tokenizer = TabularTokenizer(d_x, d_model, n_classes)
        self.layers = nn.ModuleList([
            _EncoderLayer(PureAsymAttention(d_model, n_heads, dropout),
                          d_model, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)
        self.n_classes = n_classes

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        seq = self.tokenizer(X_ctx, y_ctx, X_q)
        for layer in self.layers:
            seq = layer(seq)
        N_c = X_ctx.size(1)
        return self.head(seq[:, N_c:, :])


class StdAttentionDecomposable(nn.Module):
    """Standard separate-W_Q/W_K softmax with a runtime score-mode switch.

    ``score_mode`` is one of ``full``, ``sym``, ``asym`` and is applied to the
    pre-softmax score matrix; the value-projection path is unchanged.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.score_mode = "full"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, dh = self.n_heads, self.d_h
        q = self.W_Q(x).view(B, N, H, dh)
        k = self.W_K(x).view(B, N, H, dh)
        v = self.W_V(x).view(B, N, H, dh)
        S = torch.einsum("bnhd,bmhd->bhnm", q, k) / (dh ** 0.5)
        if self.score_mode == "sym":
            S = (S + S.transpose(-1, -2)) / 2
        elif self.score_mode == "asym":
            S = (S - S.transpose(-1, -2)) / 2
        W = F.softmax(S, dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", W, v).reshape(B, N, H * dh)
        return self.dropout(self.W_O(out))


class StdAttnDecomposableTransformer(nn.Module):
    def __init__(self, d_x: int = 8, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 1, dim_ff: int = 128, dropout: float = 0.0,
                 n_classes: int = 2) -> None:
        super().__init__()
        self.tokenizer = TabularTokenizer(d_x, d_model, n_classes)
        self.layers = nn.ModuleList([
            _EncoderLayer(StdAttentionDecomposable(d_model, n_heads, dropout),
                          d_model, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)
        self.n_classes = n_classes

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        seq = self.tokenizer(X_ctx, y_ctx, X_q)
        for layer in self.layers:
            seq = layer(seq)
        N_c = X_ctx.size(1)
        return self.head(seq[:, N_c:, :])


def set_score_mode(model: nn.Module, mode: str) -> None:
    """Set ``score_mode`` on every :class:`StdAttentionDecomposable` in ``model``."""
    for m in model.modules():
        if isinstance(m, StdAttentionDecomposable):
            m.score_mode = mode


def measure_kernel_decomposition(model: nn.Module) -> list[tuple[float, float]]:
    """Per-head ``(sym_energy_frac, asym_energy_frac)`` of ``B = W_Q^T W_K``."""
    out: list[tuple[float, float]] = []
    for m in model.modules():
        if isinstance(m, StdAttentionDecomposable):
            D = m.W_Q.weight.shape[0]
            H, dh = m.n_heads, m.d_h
            WQ = m.W_Q.weight.view(H, dh, D)
            WK = m.W_K.weight.view(H, dh, D)
            for h in range(H):
                B = WQ[h].T @ WK[h]
                B_S = (B + B.T) / 2
                B_A = (B - B.T) / 2
                tot = (B ** 2).sum().item() + 1e-12
                out.append((float((B_S ** 2).sum().item() / tot),
                            float((B_A ** 2).sum().item() / tot)))
    return out


# ----------------------------------------------------------------------------
# DualSymAsymICLTransformer (used by DualChannelAudit).
# ----------------------------------------------------------------------------


class DualSymAsymICLTransformer(nn.Module):
    """Parallel symmetric-softmax + skew-symmetric branches with summed logits."""

    def __init__(self, d_x: int = 8, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 1, dim_ff: int = 128, dropout: float = 0.0,
                 n_classes: int = 2) -> None:
        super().__init__()
        self.tokenizer = TabularTokenizer(d_x, d_model, n_classes)
        self.sym_layers = nn.ModuleList([
            _EncoderLayer(SymSoftmaxAttention(d_model, n_heads, dropout),
                          d_model, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.asym_layers = nn.ModuleList([
            _EncoderLayer(PureAsymAttention(d_model, n_heads, dropout),
                          d_model, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.sym_head = nn.Linear(d_model, n_classes)
        self.asym_head = nn.Linear(d_model, n_classes)
        self.n_classes = n_classes

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor, return_branches: bool = False):
        seq = self.tokenizer(X_ctx, y_ctx, X_q)
        sym_out = seq
        for layer in self.sym_layers:
            sym_out = layer(sym_out)
        asym_out = seq
        for layer in self.asym_layers:
            asym_out = layer(asym_out)
        N_c = X_ctx.size(1)
        sym_logits = self.sym_head(sym_out[:, N_c:, :])
        asym_logits = self.asym_head(asym_out[:, N_c:, :])
        if return_branches:
            return sym_logits + asym_logits, sym_logits, asym_logits
        return sym_logits + asym_logits


class BoostedSymPureAsym(nn.Module):
    """``logits = sym(X) + alpha * pure_asym(X)`` with the sym base frozen."""

    def __init__(self, sym_pretrained: nn.Module,
                 d_x: int = 8, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 1, dim_ff: int = 128, dropout: float = 0.0,
                 n_classes: int = 2) -> None:
        super().__init__()
        self.sym = sym_pretrained
        for p in self.sym.parameters():
            p.requires_grad = False
        self.delta = PureAsymICLTransformer(
            d_x=d_x, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_ff=dim_ff, dropout=dropout, n_classes=n_classes,
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.n_classes = n_classes

    def train(self, mode: bool = True) -> BoostedSymPureAsym:
        super().train(mode)
        self.sym.eval()
        return self

    def forward(self, X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                X_q: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sym_logits = self.sym(X_ctx, y_ctx, X_q)
        delta_logits = self.delta(X_ctx, y_ctx, X_q)
        return sym_logits + self.alpha * delta_logits


# ----------------------------------------------------------------------------
# Convenience exports.
# ----------------------------------------------------------------------------


def all_attention_module_classes() -> Iterable[type]:
    """Return all attention/model classes that should eventually move to
    ``tabkernels.attention.*`` (CH03's responsibility)."""
    return (
        KernelAttention,
        PairwisePredictor,
        StdICLTransformer,
        SymSoftmaxICLTransformer,
        PSDICLTransformer,
        PureAsymICLTransformer,
        StdAttnDecomposableTransformer,
        DualSymAsymICLTransformer,
        BoostedSymPureAsym,
    )


# Re-export commonly used pieces.
__all__ = [
    "KernelAttention",
    "PairwisePredictor",
    "TabularTokenizer",
    "StdICLTransformer",
    "PSDICLTransformer",
    "SymSoftmaxAttention",
    "SymSoftmaxICLTransformer",
    "PureAsymAttention",
    "PureAsymICLTransformer",
    "StdAttentionDecomposable",
    "StdAttnDecomposableTransformer",
    "DualSymAsymICLTransformer",
    "BoostedSymPureAsym",
    "set_score_mode",
    "measure_kernel_decomposition",
    "all_attention_module_classes",
]


# Suppress unused-numpy lint when audits import this module.
_ = np
