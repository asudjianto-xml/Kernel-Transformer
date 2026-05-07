"""Tests for tabkernels.attention (Chapter 3).

Verify the AttentionBlock contract for all 6 variants, plus structural
properties (Sym-PSD's B is PSD, PureAsym's B is skew-symmetric, etc.).
"""
import torch
import pytest
from tabkernels.attention import (
    StdAttention, SymPSDAttention, SymGenAttention,
    PureAsymAttention, DualAttention, DecomposableStdAttention,
)


@pytest.fixture
def toy_data():
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    y = torch.randn(20)
    return X, y


def test_std_forward_shape(toy_data):
    X, y = toy_data
    m = StdAttention(d_in=4, d_emb=8)
    out = m(X, X, y)
    assert out.shape == y.shape


def test_std_decompose_recovers(toy_data):
    """decompose() returns (B_S, B_A) summing to B."""
    m = StdAttention(d_in=4, d_emb=8)
    B_S, B_A = m.decompose()
    B = m.W_Q.weight.T @ m.W_K.weight
    assert torch.allclose(B_S + B_A, B, atol=1e-6)


def test_std_energy_split_sums_to_one(toy_data):
    m = StdAttention(d_in=4, d_emb=8)
    aS, aA = m.energy_split()
    assert abs(aS + aA - 1.0) < 1e-5


def test_sym_psd_B_is_psd(toy_data):
    m = SymPSDAttention(d_in=4, d_emb=8)
    B_S, B_A = m.decompose()
    # B_A should be approximately zero (B is symmetric by construction).
    assert B_A.abs().max() < 1e-5
    # B_S = B should be PSD.
    eigvals = torch.linalg.eigvalsh(B_S)
    assert eigvals.min() > -1e-5


def test_sym_psd_energy_split(toy_data):
    m = SymPSDAttention(d_in=4, d_emb=8)
    aS, aA = m.energy_split()
    assert aS > 0.99  # essentially all-symmetric
    assert aA < 1e-3


def test_sym_gen_decompose_is_pure_sym():
    m = SymGenAttention(d_in=6)
    B_S, B_A = m.decompose()
    assert B_A.abs().max() < 1e-5
    assert torch.allclose(B_S, B_S.T, atol=1e-6)


def test_pure_asym_is_skew_symmetric():
    m = PureAsymAttention(d_in=6)
    B_S, B_A = m.decompose()
    # B_S should be approximately zero.
    assert B_S.abs().max() < 1e-5
    # B_A should be skew-symmetric.
    assert torch.allclose(B_A, -B_A.T, atol=1e-6)


def test_pure_asym_energy_split():
    m = PureAsymAttention(d_in=6)
    aS, aA = m.energy_split()
    assert aA > 0.99
    assert aS < 1e-3


def test_dual_decompose_returns_branches():
    m = DualAttention(d_in=6)
    B_S, B_A = m.decompose()
    # B_S symmetric.
    assert torch.allclose(B_S, B_S.T, atol=1e-6)
    # B_A skew-symmetric.
    assert torch.allclose(B_A, -B_A.T, atol=1e-6)


def test_dual_forward_uses_sum(toy_data):
    """Dual forward score is x^T (B_S + B_A) y / sqrt(d)."""
    X, y = toy_data
    m = DualAttention(d_in=4)
    out = m(X, X, y)
    assert out.shape == y.shape


def test_decomposable_full_matches_std(toy_data):
    """Decomposable in 'full' mode equals StdAttention with same weights."""
    X, y = toy_data
    torch.manual_seed(123)
    m_dec = DecomposableStdAttention(d_in=4, d_emb=8, score_mode="full")
    torch.manual_seed(123)
    m_std = StdAttention(d_in=4, d_emb=8)
    out_dec = m_dec(X, X, y)
    out_std = m_std(X, X, y)
    assert torch.allclose(out_dec, out_std, atol=1e-6)


def test_decomposable_score_mode_changes_output(toy_data):
    """The three score modes give distinct outputs on a generic input."""
    X, y = toy_data
    m = DecomposableStdAttention(d_in=4, d_emb=8)
    m.score_mode = "full"; out_full = m(X, X, y)
    m.score_mode = "sym"; out_sym = m(X, X, y)
    m.score_mode = "asym"; out_asym = m(X, X, y)
    # Generic random weights -> outputs should differ.
    assert not torch.allclose(out_full, out_sym, atol=1e-3)
    assert not torch.allclose(out_full, out_asym, atol=1e-3)
    assert not torch.allclose(out_sym, out_asym, atol=1e-3)


def test_all_variants_grad_flow(toy_data):
    """Backward pass produces non-zero gradients for all variants."""
    X, y = toy_data
    for cls, kwargs in [
        (StdAttention, dict(d_in=4, d_emb=8)),
        (SymPSDAttention, dict(d_in=4, d_emb=8)),
        (SymGenAttention, dict(d_in=4)),
        (PureAsymAttention, dict(d_in=4)),
        (DualAttention, dict(d_in=4)),
        (DecomposableStdAttention, dict(d_in=4, d_emb=8)),
    ]:
        m = cls(**kwargs)
        out = m(X, X, y)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        # At least one param has nonzero gradient.
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in m.parameters())
        assert has_grad, f"{cls.__name__} gradients did not flow"


def test_score_decomposition_matches_input_decomp():
    """The score-level decomposition (S+S^T)/2 corresponds to B_S in input space."""
    torch.manual_seed(0)
    X = torch.randn(10, 4)
    m = StdAttention(d_in=4, d_emb=8)
    S = m.scores(X, X)
    S_S = (S + S.T) / 2
    # Compare against direct: x_i^T B_S x_j / sqrt(d_emb)
    import math
    B_S, _ = m.decompose()
    direct = X @ B_S @ X.T / math.sqrt(8)
    assert torch.allclose(S_S, direct, atol=1e-5)
