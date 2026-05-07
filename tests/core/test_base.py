"""Tests that lock the base-class API contract.

These tests verify the *contract* — they do not test specific implementations.
Chapter agents add their own tests under tests/<module>/.
"""
import pytest
import torch

from tabkernels.core.base import (
    Architecture,
    AttentionBlock,
    Audit,
    Kernel,
    Predictor,
    Prior,
    Sparsifier,
    Symmetrizer,
)
from tabkernels.core.decomposition import (
    cosine_similarity_F,
    decompose,
    energy_split,
    project_score,
)

# ----------- decomposition primitives -----------

def test_decompose_orthogonality():
    """Frobenius orthogonality: <B_S, B_A>_F = 0."""
    torch.manual_seed(0)
    B = torch.randn(8, 8)
    B_S, B_A = decompose(B)
    inner = (B_S * B_A).sum().item()
    assert abs(inner) < 1e-5


def test_decompose_recovers_B():
    """B_S + B_A = B."""
    torch.manual_seed(0)
    B = torch.randn(8, 8)
    B_S, B_A = decompose(B)
    assert torch.allclose(B_S + B_A, B, atol=1e-6)


def test_decompose_symmetric_input():
    """For symmetric B, B_A = 0."""
    torch.manual_seed(0)
    M = torch.randn(8, 8)
    B = (M + M.T) / 2
    _, B_A = decompose(B)
    assert B_A.abs().max().item() < 1e-6


def test_energy_split_sums_to_one():
    torch.manual_seed(0)
    B = torch.randn(8, 8)
    s, a = energy_split(B)
    assert abs(s + a - 1.0) < 1e-5


def test_cosine_self_is_one():
    torch.manual_seed(0)
    A = torch.randn(8, 8)
    assert abs(cosine_similarity_F(A, A) - 1.0) < 1e-5


def test_project_score_modes():
    torch.manual_seed(0)
    S = torch.randn(4, 4)
    assert torch.allclose(project_score(S, "full"), S)
    sym = project_score(S, "sym")
    assert torch.allclose(sym, sym.T, atol=1e-6)
    asym = project_score(S, "asym")
    assert torch.allclose(asym, -asym.T, atol=1e-6)
    with pytest.raises(ValueError):
        project_score(S, "nonsense")


# ----------- base class abstract enforcement -----------

def test_kernel_must_implement_forward():
    """Cannot instantiate Kernel without forward()."""
    with pytest.raises(TypeError):
        Kernel()


def test_sparsifier_must_implement_forward():
    with pytest.raises(TypeError):
        Sparsifier()


def test_symmetrizer_must_implement_forward():
    with pytest.raises(TypeError):
        Symmetrizer()


def test_predictor_must_implement_forward():
    with pytest.raises(TypeError):
        Predictor()


def test_attention_block_must_implement_forward_and_decompose():
    with pytest.raises(TypeError):
        AttentionBlock()


def test_architecture_must_implement_forward():
    with pytest.raises(TypeError):
        Architecture()


def test_prior_must_implement_sample_episode():
    with pytest.raises(TypeError):
        Prior()


def test_audit_must_implement_run():
    with pytest.raises(TypeError):
        Audit()
