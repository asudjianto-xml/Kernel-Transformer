"""Tests for tabkernels.architectures.tabpfn_lite (Chapter 16)."""
from __future__ import annotations

import pytest
import torch

from tabkernels.architectures import TabPFNLite
from tabkernels.audits._models import StdAttentionDecomposable


def test_construction_validates_head_divisor():
    with pytest.raises(ValueError):
        TabPFNLite(d_in=4, d_model=15, n_heads=4)


def test_forward_unbatched_shape():
    m = TabPFNLite(d_in=4, d_model=32, n_heads=4, n_layers=2)
    X_ctx = torch.randn(20, 4); y_ctx = torch.randn(20)
    X_q = torch.randn(8, 4)
    out = m(X_q, X_ctx, y_ctx)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_forward_batched_shape():
    m = TabPFNLite(d_in=4, d_model=32, n_heads=4, n_layers=2)
    X_ctx = torch.randn(2, 20, 4); y_ctx = torch.randn(2, 20)
    X_q = torch.randn(2, 8, 4)
    out = m(X_q, X_ctx, y_ctx)
    assert out.shape == (2, 8)


def test_attention_blocks_returns_n_layers():
    m = TabPFNLite(d_in=4, d_model=32, n_heads=4, n_layers=3)
    blocks = m.attention_blocks()
    assert len(blocks) == 3
    for b in blocks:
        assert isinstance(b, StdAttentionDecomposable)


def test_decompose_attention_recognises_tabpfn_blocks():
    """Sanity check that the CH13 decomposition diagnostic accepts TabPFNLite."""
    from tabkernels.transparency import decompose_attention
    m = TabPFNLite(d_in=4, d_model=32, n_heads=4, n_layers=2)
    for block in m.attention_blocks():
        out = decompose_attention(block)
        assert "B" in out and "B_S" in out and "B_A" in out
        # Per-head extraction stacks heads on dim 0.
        assert out["B"].shape[-1] == out["B"].shape[-2]


def test_kernel_energy_split_per_block():
    """Frobenius energy of B_S and B_A should sum to ~1 per head."""
    from tabkernels.core.decomposition import energy_split
    from tabkernels.transparency import decompose_attention
    m = TabPFNLite(d_in=4, d_model=32, n_heads=4, n_layers=2)
    for block in m.attention_blocks():
        d = decompose_attention(block)
        for h in range(d["B"].shape[0]):
            a_s, a_a = energy_split(d["B"][h])
            assert abs(a_s + a_a - 1.0) < 1e-3


def test_train_step_reduces_loss():
    """A short PFN-style training run should reduce loss."""
    torch.manual_seed(0)
    m = TabPFNLite(d_in=4, d_model=32, n_heads=4, n_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        X_ctx = torch.randn(24, 4); X_q = torch.randn(8, 4)
        w = torch.randn(4)
        y_ctx = X_ctx @ w + 0.1 * torch.randn(24)
        y_q = X_q @ w + 0.1 * torch.randn(8)
        opt.zero_grad()
        yhat = m(X_q, X_ctx, y_ctx)
        loss = ((yhat - y_q) ** 2).mean()
        loss.backward(); opt.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_subclasses_architecture_base():
    from tabkernels.core.base import Architecture
    assert issubclass(TabPFNLite, Architecture)
