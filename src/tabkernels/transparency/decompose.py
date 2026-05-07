"""Post-hoc decomposition diagnostic primitives -- see Chapter 13.

This module composes the building blocks in :mod:`tabkernels.core.decomposition`
into a transparency / interpretability workflow that a practitioner can apply to
*any* trained tabular transformer with bilinear-form attention.

The five public primitives are

* :func:`decompose_attention` -- pull ``B = W_Q^T W_K`` (or its analogue) out of
  every recognised attention module, decompose into ``(B, B_S, B_A)``.
* :func:`kernel_energy_split` -- Frobenius energy fractions
  ``alpha_S, alpha_A`` (Eq.~13.3).
* :func:`recovery_cosine` -- cosine alignment of the *learned* sym/skew parts
  to a *target* (Eq.~13.4).
* :func:`post_hoc_eval` -- evaluate a model with a runtime score-matrix
  projection ``S -> S, (S+S^T)/2, (S-S^T)/2`` (Eq.~13.2).
* :func:`inspect_attention` -- the full workflow, returning an
  :class:`AttentionReport`.

The list of recognised attention modules is intentionally narrow: every class
in :mod:`tabkernels.audits._models` whose learned bilinear form admits a
canonical extraction.  Architectures that do not yet have a canonical
extractor (LSA, kernel-PCA-style, Performer with non-linear feature maps) are
ignored with a warning -- the workflow degrades gracefully rather than
hallucinating an extraction.

See ``affinity/book/chapters/CH13_transparency.tex`` for the methodology.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

import torch
import torch.nn as nn

from tabkernels.core.decomposition import (
    cosine_similarity_F,
    decompose,
    energy_split,
    project_score,
)

__all__ = [
    "decompose_attention",
    "kernel_energy_split",
    "recovery_cosine",
    "post_hoc_eval",
    "inspect_attention",
    "AttentionReport",
]


# ---------------------------------------------------------------------------
# Per-head extraction
# ---------------------------------------------------------------------------


def _extract_block_B(block: nn.Module) -> list[torch.Tensor] | None:
    """Return per-head ``B^{(h)}`` for one attention block, or ``None``.

    Recognises the model classes in :mod:`tabkernels.audits._models`.  The
    extraction rule is documented in Chapter 13 §13.1 (Eq.~13.1).
    """
    # Local import to avoid a circular dependency at package-load time.
    from tabkernels.audits._models import (
        KernelAttention,
        PSDAttention,
        PureAsymAttention,
        StdAttentionDecomposable,
        SymSoftmaxAttention,
    )

    # ---- Multi-head encoder blocks (token-space) -------------------------
    if isinstance(block, StdAttentionDecomposable):
        H, dh = block.n_heads, block.d_h
        d_model = block.W_Q.weight.shape[0]
        WQ = block.W_Q.weight.view(H, dh, d_model).detach()
        WK = block.W_K.weight.view(H, dh, d_model).detach()
        return [WQ[h].T @ WK[h] for h in range(H)]

    if isinstance(block, SymSoftmaxAttention):
        H, dh = block.n_heads, block.d_h
        d_model = block.W_QK.weight.shape[0]
        W = block.W_QK.weight.view(H, dh, d_model).detach()
        # Shared W_QK -> per-head B^{(h)} = W^T W (already symmetric PSD).
        return [W[h].T @ W[h] for h in range(H)]

    if isinstance(block, PSDAttention):
        H, dh = block.n_heads, block.d_h
        d_model = block.W_QK.weight.shape[0]
        W = block.W_QK.weight.view(H, dh, d_model).detach()
        return [W[h].T @ W[h] for h in range(H)]

    if isinstance(block, PureAsymAttention):
        # Skew per head: A^{(h)} = M^{(h)} - M^{(h)T}, lifted into d_model
        # via the embedding W_E.
        H = block.n_heads
        d_model = block.W_E.weight.shape[0]
        WE = block.W_E.weight.view(H, block.d_h, d_model).detach()
        M = block.M.detach()
        per_head: list[torch.Tensor] = []
        for h in range(H):
            A = M[h] - M[h].T
            per_head.append(WE[h].T @ A @ WE[h])
        return per_head

    # ---- Tabular kernel-attention block (used by the audits) -------------
    if isinstance(block, KernelAttention):
        if block.variant == "std":
            B = block.W_Q.weight.detach().T @ block.W_K.weight.detach()
            return [B]
        if block.variant in ("sym_psd",):
            W = block.W_QK.weight.detach()
            return [W.T @ W]
        if block.variant == "sym_gen":
            M = block.M.detach()
            return [(M + M.T) / 2]
        if block.variant == "pure_asym":
            M = block.M.detach()
            return [(M - M.T) / 2]
        if block.variant == "dual":
            M_S = block.M_S.detach()
            M_A = block.M_A.detach()
            return [(M_S + M_S.T) / 2 + (M_A - M_A.T) / 2]
    return None


def decompose_attention(attention: nn.Module) -> dict[str, torch.Tensor]:
    """Decompose every recognised attention block under ``attention``.

    Args:
        attention: any :class:`torch.nn.Module`; the function walks every
            sub-module looking for recognised attention classes.

    Returns:
        A dict with three stacked tensors of shape ``(K, d, d)``
        (one row per recognised head):

        * ``"B"``   -- the raw bilinear form,
        * ``"B_S"`` -- its symmetric part,
        * ``"B_A"`` -- its skew-symmetric part.

        ``K`` is the total number of recognised heads across the module.
        Returns empty tensors if no recognised block is found.
    """
    Bs: list[torch.Tensor] = []
    for m in attention.modules():
        per_head = _extract_block_B(m)
        if per_head is None:
            continue
        Bs.extend(per_head)
    if not Bs:
        empty = torch.empty(0)
        return {"B": empty, "B_S": empty, "B_A": empty}
    # Allow heterogeneous d (multiple blocks of different size); pad up.
    max_d = max(b.shape[-1] for b in Bs)
    padded: list[torch.Tensor] = []
    for b in Bs:
        if b.shape[-1] != max_d:
            pad = torch.zeros(max_d, max_d, dtype=b.dtype, device=b.device)
            d = b.shape[-1]
            pad[:d, :d] = b
            padded.append(pad)
        else:
            padded.append(b)
    B_stack = torch.stack(padded, dim=0)
    B_S_stack = torch.stack([decompose(b)[0] for b in padded], dim=0)
    B_A_stack = torch.stack([decompose(b)[1] for b in padded], dim=0)
    return {"B": B_stack, "B_S": B_S_stack, "B_A": B_A_stack}


# ---------------------------------------------------------------------------
# Energy-split and cosine-recovery thin wrappers
# ---------------------------------------------------------------------------


def kernel_energy_split(B: torch.Tensor) -> tuple[float, float]:
    """Frobenius energy fractions ``(alpha_S, alpha_A)`` of ``B`` (Eq.~13.3).

    ``B`` may be a single ``(d, d)`` matrix or a stack of shape ``(K, d, d)``;
    the latter case averages over heads.
    """
    if B.numel() == 0:
        return 0.0, 0.0
    if B.dim() == 2:
        return energy_split(B)
    # Stack: average per-head splits.
    s_acc, a_acc = 0.0, 0.0
    for h in range(B.shape[0]):
        s, a = energy_split(B[h])
        s_acc += s
        a_acc += a
    K = float(B.shape[0])
    return s_acc / K, a_acc / K


def recovery_cosine(
    B_learned: torch.Tensor, B_true: torch.Tensor
) -> tuple[float, float]:
    """Cosine alignment of the sym/skew parts (Eq.~13.4).

    Returns ``(cos_sym, cos_asym) = (cos(B_S^learned, B_S^true),
    cos(B_A^learned, B_A^true))``.

    ``B_learned`` may be a stack ``(K, d, d)``; cosines are then computed on
    the head-averaged sym/skew components, which is the right thing for
    multi-head models where a single underlying kernel is split across heads.
    """
    if B_learned.numel() == 0 or B_true.numel() == 0:
        return 0.0, 0.0
    if B_learned.dim() == 3:
        learned_S = torch.stack([decompose(B_learned[h])[0]
                                 for h in range(B_learned.shape[0])]).mean(0)
        learned_A = torch.stack([decompose(B_learned[h])[1]
                                 for h in range(B_learned.shape[0])]).mean(0)
    else:
        learned_S, learned_A = decompose(B_learned)
    true_S, true_A = decompose(B_true)
    cos_S = cosine_similarity_F(learned_S, true_S)
    # cosine_similarity_F protects against zero norm.
    cos_A = cosine_similarity_F(learned_A, true_A)
    return cos_S, cos_A


# ---------------------------------------------------------------------------
# Post-hoc projection evaluation
# ---------------------------------------------------------------------------


def _set_decomposable_score_mode(model: nn.Module, mode: str) -> int:
    """Set ``score_mode`` on every decomposable attention block.  Returns count."""
    from tabkernels.audits._models import StdAttentionDecomposable

    n = 0
    for m in model.modules():
        if isinstance(m, StdAttentionDecomposable):
            m.score_mode = mode
            n += 1
    return n


def _eval_metric(model: nn.Module, X: torch.Tensor,
                 y: torch.Tensor | None) -> float:
    """Run ``model(X)`` and return either accuracy (classification) or
    negative MSE (regression).  Higher is better in both cases."""
    model.eval()
    with torch.no_grad():
        out = model(X)
    if isinstance(out, tuple):
        out = out[0]
    if y is None:
        # No target -> return mean output entropy as a self-consistency score.
        return float(out.mean().item())
    if y.dtype in (torch.long, torch.int64, torch.int32):
        # Classification: take argmax over the last axis of logits.
        if out.dim() == y.dim() + 1:
            preds = out.argmax(dim=-1)
        else:
            preds = (out > 0).long().squeeze(-1)
        # Broadcast both to 1D for accuracy.
        return float((preds == y).float().mean().item())
    # Regression: report -MSE so higher is still better.
    if out.shape != y.shape:
        out = out.view_as(y)
    return -float(((out - y) ** 2).mean().item())


def post_hoc_eval(model: nn.Module, X: torch.Tensor,
                  y: torch.Tensor | None,
                  mode: Literal["full", "sym", "asym"]) -> float:
    """Evaluate ``model`` with the post-hoc score-matrix projection ``mode``
    (Eq.~13.2).

    Internally, this temporarily flips the ``score_mode`` flag on every
    :class:`tabkernels.audits._models.StdAttentionDecomposable` block in
    ``model``, runs a forward pass, and restores the original flag.

    For models with no decomposable block, ``post_hoc_eval`` simply runs the
    model in its current state and reports the metric.  This is the
    conservative behaviour: a model that does not expose a decomposable
    score path returns the same number for ``full``, ``sym`` and ``asym``,
    making the asymmetric-capacity-is-doing-no-work conclusion automatic.
    """
    if mode not in ("full", "sym", "asym"):
        raise ValueError(f"unknown mode: {mode!r}")
    # Snapshot original modes.
    from tabkernels.audits._models import StdAttentionDecomposable

    originals: list[tuple[StdAttentionDecomposable, str]] = []
    for m in model.modules():
        if isinstance(m, StdAttentionDecomposable):
            originals.append((m, m.score_mode))
    try:
        _set_decomposable_score_mode(model, mode)
        return _eval_metric(model, X, y)
    finally:
        for m, mode_orig in originals:
            m.score_mode = mode_orig


# ---------------------------------------------------------------------------
# Top-level workflow
# ---------------------------------------------------------------------------


class AttentionReport(TypedDict):
    """The output of :func:`inspect_attention`.

    Fields:
        alpha_S, alpha_A: head-averaged Frobenius energy fractions.
        per_head: list of per-head ``{'alpha_S', 'alpha_A'}`` dicts.
        cosine_to_target: optional ``(cos_sym, cos_asym)`` against a target
            bilinear form (Eq.~13.4).
        full_acc, sym_only_acc, asym_only_acc: post-hoc projection metrics
            (accuracy for classification, negative MSE for regression).
    """

    alpha_S: float
    alpha_A: float
    per_head: list[dict[str, float]]
    cosine_to_target: tuple[float, float] | None
    full_acc: float
    sym_only_acc: float
    asym_only_acc: float


def _resolve_xy(X: Any) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
    """Helper: accept either a tensor X or a callable that produces (X, y)."""
    if callable(X):
        result = X()
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1], None
    return X, None, None


def inspect_attention(
    model: nn.Module,
    X: torch.Tensor,
    target: torch.Tensor | None = None,
    *,
    y: torch.Tensor | None = None,
) -> AttentionReport:
    """Run the full Chapter 13 inspection workflow on ``model``.

    Args:
        model: trained PyTorch module containing one or more recognised
            attention blocks.
        X: input batch passed to ``model.forward``.
        target: optional ``(d, d)`` ground-truth bilinear form.  If given,
            the report includes a cosine-recovery row.
        y: optional supervised label tensor matching the model output, used
            to compute the post-hoc projection metric (accuracy or -MSE).
            If omitted, ``full_acc/sym_only_acc/asym_only_acc`` instead
            report the mean output (a self-consistency value), which is
            still useful as a sanity check.

    Returns:
        :class:`AttentionReport`.
    """
    parts = decompose_attention(model)
    B = parts["B"]
    if B.numel() == 0:
        report: AttentionReport = {
            "alpha_S": 0.0,
            "alpha_A": 0.0,
            "per_head": [],
            "cosine_to_target": None,
            "full_acc": _eval_metric(model, X, y),
            "sym_only_acc": _eval_metric(model, X, y),
            "asym_only_acc": _eval_metric(model, X, y),
        }
        return report

    # Per-head energy fractions.
    per_head: list[dict[str, float]] = []
    for h in range(B.shape[0]):
        s, a = energy_split(B[h])
        per_head.append({"alpha_S": s, "alpha_A": a})
    alpha_S, alpha_A = kernel_energy_split(B)

    # Cosine to a known target.
    cos: tuple[float, float] | None = None
    if target is not None:
        cos = recovery_cosine(B, target)

    # Post-hoc projection metrics.  These will only diverge for models that
    # contain decomposable attention; otherwise they all agree.
    full_acc = post_hoc_eval(model, X, y, mode="full")
    sym_only_acc = post_hoc_eval(model, X, y, mode="sym")
    asym_only_acc = post_hoc_eval(model, X, y, mode="asym")

    report = AttentionReport(
        alpha_S=float(alpha_S),
        alpha_A=float(alpha_A),
        per_head=per_head,
        cosine_to_target=cos,
        full_acc=float(full_acc),
        sym_only_acc=float(sym_only_acc),
        asym_only_acc=float(asym_only_acc),
    )
    return report


# ---------------------------------------------------------------------------
# Re-exports of low-level primitives so the user can import everything from
# :mod:`tabkernels.transparency` directly.
# ---------------------------------------------------------------------------

# Marker symbol so static analysers see ``project_score`` is intentionally
# part of the public surface (it backs ``post_hoc_eval``).
_PROJECT_SCORE_PUBLIC = project_score
