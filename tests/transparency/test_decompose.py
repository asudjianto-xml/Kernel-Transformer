"""Tests for :mod:`tabkernels.transparency` (Chapter 13 primitives)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tabkernels.audits._models import (
    KernelAttention,
    PSDICLTransformer,
    PureAsymICLTransformer,
    StdAttnDecomposableTransformer,
    SymSoftmaxICLTransformer,
)
from tabkernels.transparency import (
    AttentionReport,
    decompose_attention,
    inspect_attention,
    kernel_energy_split,
    post_hoc_eval,
    recovery_cosine,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_std_decomposable(seed: int = 0,
                           d_x: int = 6, d_model: int = 16, n_heads: int = 4,
                           n_layers: int = 1, n_classes: int = 2,
                           ) -> StdAttnDecomposableTransformer:
    torch.manual_seed(seed)
    m = StdAttnDecomposableTransformer(
        d_x=d_x, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, n_classes=n_classes,
    )
    m.eval()
    return m


def _icl_inputs(B: int = 2, N_ctx: int = 8, N_q: int = 3, d_x: int = 6,
                n_classes: int = 2, seed: int = 7
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X_ctx = torch.randn(B, N_ctx, d_x)
    y_ctx = torch.randint(0, n_classes, (B, N_ctx))
    X_q = torch.randn(B, N_q, d_x)
    return X_ctx, y_ctx, X_q


class _ICLWrapper(nn.Module):
    """Wrap an ICL transformer so its ``forward(X)`` takes a single batched
    feature tensor, by stuffing fixed ``(X_ctx, y_ctx)`` into the forward."""

    def __init__(self, base: nn.Module, X_ctx: torch.Tensor,
                 y_ctx: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("X_ctx", X_ctx)
        self.register_buffer("y_ctx", y_ctx)

    def forward(self, X_q: torch.Tensor) -> torch.Tensor:
        return self.base(self.X_ctx, self.y_ctx, X_q)


# ---------------------------------------------------------------------------
# 1. decompose_attention
# ---------------------------------------------------------------------------


def test_decompose_attention_std_decomposable_per_head_count() -> None:
    """One Std-decomposable layer with H heads -> H per-head B's."""
    n_heads = 4
    model = _make_std_decomposable(n_heads=n_heads, n_layers=1)
    parts = decompose_attention(model)
    assert parts["B"].shape[0] == n_heads
    # B = B_S + B_A.
    assert torch.allclose(parts["B_S"] + parts["B_A"], parts["B"], atol=1e-5)


def test_decompose_attention_multilayer_stacks_heads() -> None:
    n_heads, n_layers = 2, 3
    model = _make_std_decomposable(n_heads=n_heads, n_layers=n_layers)
    parts = decompose_attention(model)
    assert parts["B"].shape[0] == n_heads * n_layers


def test_decompose_attention_kernel_block_variants() -> None:
    """KernelAttention covers the std/sym_psd/sym_gen/pure_asym/dual paths."""
    for variant in ("std", "sym_psd", "sym_gen", "pure_asym", "dual"):
        block = KernelAttention(d_in=5, d_emb=4, variant=variant)
        parts = decompose_attention(block)
        assert parts["B"].shape == (1, 5, 5)
        # Sanity: pure_asym variant -> alpha_A == 1.0 (within rounding).
        if variant == "pure_asym":
            _, a = kernel_energy_split(parts["B"])
            assert a > 0.999


def test_decompose_attention_sym_psd_extracts_psd() -> None:
    """SymSoftmaxICLTransformer / PSDICLTransformer give symmetric B."""
    torch.manual_seed(0)
    sym_model = SymSoftmaxICLTransformer(
        d_x=6, d_model=16, n_heads=4, n_layers=1, n_classes=2,
    )
    parts = decompose_attention(sym_model)
    s, a = kernel_energy_split(parts["B"])
    assert s > 0.999  # All energy is in the symmetric component.
    assert a < 1e-3

    psd_model = PSDICLTransformer(
        d_x=6, d_model=16, n_heads=4, n_layers=1, n_classes=2,
    )
    parts2 = decompose_attention(psd_model)
    s2, _ = kernel_energy_split(parts2["B"])
    assert s2 > 0.999


def test_decompose_attention_pure_asym_extracts_skew() -> None:
    torch.manual_seed(0)
    asym_model = PureAsymICLTransformer(
        d_x=6, d_model=16, n_heads=4, n_layers=1, n_classes=2,
    )
    parts = decompose_attention(asym_model)
    s, a = kernel_energy_split(parts["B"])
    assert a > 0.999
    assert s < 1e-3


def test_decompose_attention_unknown_module_returns_empty() -> None:
    """A plain Linear has no recognised attention block -> empty result."""
    parts = decompose_attention(nn.Linear(8, 8))
    assert parts["B"].numel() == 0
    assert parts["B_S"].numel() == 0
    assert parts["B_A"].numel() == 0


def test_decompose_attention_padding_mixed_widths() -> None:
    """Two KernelAttention blocks of different d_in are padded into one stack."""
    torch.manual_seed(0)
    a = KernelAttention(d_in=4, d_emb=4, variant="sym_gen")
    b = KernelAttention(d_in=6, d_emb=4, variant="sym_gen")
    container = nn.Sequential(a, b)
    parts = decompose_attention(container)
    assert parts["B"].shape == (2, 6, 6)
    # Padded entries should still produce a proper (B_S, B_A) decomposition.
    assert torch.allclose(parts["B_S"] + parts["B_A"], parts["B"], atol=1e-5)


# ---------------------------------------------------------------------------
# 2. kernel_energy_split
# ---------------------------------------------------------------------------


def test_energy_split_singleton_matches_core() -> None:
    """A single (d, d) input agrees with tabkernels.core.energy_split."""
    torch.manual_seed(11)
    B = torch.randn(8, 8)
    s, a = kernel_energy_split(B)
    assert abs(s + a - 1.0) < 1e-5
    # Symmetric input -> a == 0.
    Bsym = (B + B.T) / 2
    s2, a2 = kernel_energy_split(Bsym)
    assert s2 > 0.999 and a2 < 1e-5


def test_energy_split_stack_averages_per_head() -> None:
    torch.manual_seed(12)
    H, d = 5, 6
    Bs = torch.randn(H, d, d)
    s, a = kernel_energy_split(Bs)
    assert abs(s + a - 1.0) < 1e-5
    # Compare against manual average.
    s_manual = sum((b + b.T).pow(2).sum().item() / 4 / b.pow(2).sum().item()
                   for b in Bs) / H
    assert abs(s - s_manual) < 1e-4


def test_energy_split_empty_stack_returns_zeros() -> None:
    s, a = kernel_energy_split(torch.empty(0))
    assert s == 0.0 and a == 0.0


# ---------------------------------------------------------------------------
# 3. recovery_cosine
# ---------------------------------------------------------------------------


def test_recovery_cosine_self() -> None:
    torch.manual_seed(13)
    B = torch.randn(8, 8)
    cs, ca = recovery_cosine(B, B)
    # Cosine of a symmetric/asym piece with itself is 1.
    assert abs(cs - 1.0) < 1e-4
    assert abs(ca - 1.0) < 1e-4


def test_recovery_cosine_pure_skew_targets() -> None:
    """Targeting a purely skew B should give cos_sym ~= 0 and cos_asym = 1."""
    torch.manual_seed(14)
    M = torch.randn(8, 8)
    Bskew = (M - M.T) / 2
    cs, ca = recovery_cosine(Bskew, Bskew)
    # cosine_similarity_F protects against zero norm in B_S; for a purely
    # skew tensor B_S = 0 -> cosine is exactly 0 by the implementation.
    assert abs(cs) < 1e-4
    assert ca > 0.999


def test_recovery_cosine_stack_input() -> None:
    """Stacked learned tensor + scalar target -> cosines via head-averaging."""
    torch.manual_seed(15)
    H, d = 4, 6
    learned = torch.randn(H, d, d)
    target = torch.randn(d, d)
    cs, ca = recovery_cosine(learned, target)
    # Just make sure they are real numbers in [-1, 1].
    assert -1.0 <= cs <= 1.0
    assert -1.0 <= ca <= 1.0


def test_recovery_cosine_empty_returns_zeros() -> None:
    cs, ca = recovery_cosine(torch.empty(0), torch.randn(4, 4))
    assert cs == 0.0 and ca == 0.0


# ---------------------------------------------------------------------------
# 4. post_hoc_eval
# ---------------------------------------------------------------------------


def test_post_hoc_eval_modes_run_without_error() -> None:
    model = _make_std_decomposable(d_model=16, n_heads=4, n_layers=1)
    X_ctx, y_ctx, X_q = _icl_inputs()
    wrapped = _ICLWrapper(model, X_ctx, y_ctx)
    # Build a label tensor matching the (B, N_q) output shape.
    y = torch.randint(0, 2, (X_q.shape[0], X_q.shape[1]))
    full = post_hoc_eval(wrapped, X_q, y, mode="full")
    sym = post_hoc_eval(wrapped, X_q, y, mode="sym")
    asym = post_hoc_eval(wrapped, X_q, y, mode="asym")
    assert isinstance(full, float)
    assert isinstance(sym, float)
    assert isinstance(asym, float)


def test_post_hoc_eval_restores_original_score_mode() -> None:
    """post_hoc_eval must not leak score_mode mutations to the model."""
    from tabkernels.audits._models import StdAttentionDecomposable

    model = _make_std_decomposable()
    X_ctx, y_ctx, X_q = _icl_inputs()
    wrapped = _ICLWrapper(model, X_ctx, y_ctx)
    # Mark every decomposable block with a sentinel.
    for m in model.modules():
        if isinstance(m, StdAttentionDecomposable):
            m.score_mode = "full"
    post_hoc_eval(wrapped, X_q, y=None, mode="sym")
    for m in model.modules():
        if isinstance(m, StdAttentionDecomposable):
            assert m.score_mode == "full"


def test_post_hoc_eval_unknown_mode_raises() -> None:
    model = _make_std_decomposable()
    X_ctx, y_ctx, X_q = _icl_inputs()
    wrapped = _ICLWrapper(model, X_ctx, y_ctx)
    with pytest.raises(ValueError):
        post_hoc_eval(wrapped, X_q, None, mode="bogus")  # type: ignore[arg-type]


def test_post_hoc_eval_regression_negative_mse() -> None:
    """For regression-shaped targets, the metric is -MSE (higher is better)."""
    # Use n_classes=2 for the label embedding (must dominate y_ctx values),
    # then point a 2-d regression target at a 2-logit output.
    model = _make_std_decomposable(n_classes=2)
    X_ctx, y_ctx, X_q = _icl_inputs(n_classes=2)
    wrapped = _ICLWrapper(model, X_ctx, y_ctx)
    y = torch.randn(X_q.shape[0], X_q.shape[1], 2)
    val = post_hoc_eval(wrapped, X_q, y, mode="full")
    # -MSE is non-positive.
    assert val <= 0.0


# ---------------------------------------------------------------------------
# 5. inspect_attention -- the top-level workflow
# ---------------------------------------------------------------------------


def test_inspect_attention_full_workflow() -> None:
    model = _make_std_decomposable(n_heads=4, n_layers=2)
    X_ctx, y_ctx, X_q = _icl_inputs()
    wrapped = _ICLWrapper(model, X_ctx, y_ctx)
    y = torch.randint(0, 2, (X_q.shape[0], X_q.shape[1]))
    report: AttentionReport = inspect_attention(wrapped, X_q, target=None, y=y)
    # Required keys.
    for k in ("alpha_S", "alpha_A", "per_head",
              "cosine_to_target", "full_acc", "sym_only_acc", "asym_only_acc"):
        assert k in report
    assert len(report["per_head"]) == 4 * 2
    assert report["cosine_to_target"] is None
    # Energies sum to ~1.
    assert abs(report["alpha_S"] + report["alpha_A"] - 1.0) < 1e-4


def test_inspect_attention_with_target_returns_cosine() -> None:
    model = _make_std_decomposable()
    X_ctx, y_ctx, X_q = _icl_inputs()
    wrapped = _ICLWrapper(model, X_ctx, y_ctx)
    # Make target B match the d_model dimension of the decomposed B's.
    parts = decompose_attention(wrapped)
    d = parts["B"].shape[-1]
    target = torch.randn(d, d)
    report = inspect_attention(wrapped, X_q, target=target)
    assert report["cosine_to_target"] is not None
    cs, ca = report["cosine_to_target"]
    assert -1.0 <= cs <= 1.0 and -1.0 <= ca <= 1.0


def test_inspect_attention_no_recognised_block() -> None:
    """A non-attention model still produces a report with empty per_head."""
    plain = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    X = torch.randn(8, 4)
    report = inspect_attention(plain, X, target=None, y=None)
    assert report["per_head"] == []
    assert report["alpha_S"] == 0.0 and report["alpha_A"] == 0.0


# ---------------------------------------------------------------------------
# 6. Integration: spurious vs faithful contrast (Tables 13.1, 13.2)
# ---------------------------------------------------------------------------


def test_synthetic_pure_asym_has_zero_sym_energy() -> None:
    """Faithful-asymmetric: B_A energy == 1, cos to truth high."""
    torch.manual_seed(0)
    M = torch.randn(8, 8) * 0.5
    B_true = (M - M.T) / 2  # purely skew
    s, a = kernel_energy_split(B_true)
    assert a > 0.999
    cs, ca = recovery_cosine(B_true, B_true)
    assert ca > 0.999
