"""Tests for tabkernels.composite (Chapter 9)."""
import torch
import pytest
from tabkernels.composite import MultiHead, ChunkMultiHead, MultiScale, MultiKernel
from tabkernels.core.base import Kernel


class _ConstantKernel(Kernel):
    """Test helper: returns a constant Gram matrix."""
    def __init__(self, value: float):
        super().__init__()
        self.value = value
    def forward(self, X1, X2):
        return torch.full((X1.shape[0], X2.shape[0]), self.value, dtype=X1.dtype)


def test_multihead_average():
    """MultiHead averages per-head outputs."""
    h1 = _ConstantKernel(1.0); h2 = _ConstantKernel(3.0)
    m = MultiHead([h1, h2])
    X = torch.randn(5, 4)
    out = m(X, X)
    assert torch.allclose(out, torch.full((5, 5), 2.0))


def test_chunk_multihead_split():
    """ChunkMultiHead splits input into H chunks of d/H."""
    factory = lambda d: _ConstantKernel(1.0)
    m = ChunkMultiHead(factory, H=4, d_in=8)
    assert m.d_chunk == 2
    X = torch.randn(5, 8)
    out = m(X, X)
    assert torch.allclose(out, torch.ones(5, 5))


def test_chunk_multihead_invalid_split():
    factory = lambda d: _ConstantKernel(1.0)
    with pytest.raises(ValueError):
        ChunkMultiHead(factory, H=3, d_in=10)


def test_multiscale_simplex_weights():
    """Mixing weights live on the simplex."""
    m = MultiScale(bandwidths=[0.1, 1.0, 10.0])
    import torch.nn.functional as F
    beta = F.softmax(m.beta_logits, dim=0)
    assert abs(beta.sum().item() - 1.0) < 1e-5
    assert (beta >= 0).all()


def test_multiscale_psd():
    """MultiScale RBF is PSD."""
    torch.manual_seed(0)
    X = torch.randn(8, 3)
    m = MultiScale(bandwidths=[0.5, 1.0, 2.0])
    K = m(X, X)
    eigvals = torch.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-5


def test_multiscale_diagonal_one():
    """Self-similarity at a single point is 1 (RBF at zero distance)."""
    X = torch.tensor([[0.0, 0.0]])
    m = MultiScale(bandwidths=[1.0, 2.0])
    K = m(X, X)
    assert abs(K.item() - 1.0) < 1e-5


def test_multikernel_convex_combination():
    """MultiKernel weights are convex (softmax)."""
    k1 = _ConstantKernel(1.0); k2 = _ConstantKernel(0.0)
    m = MultiKernel([k1, k2])
    X = torch.randn(5, 4)
    out = m(X, X)
    # Default beta_logits = 0 -> uniform weights -> 0.5 * 1 + 0.5 * 0 = 0.5.
    assert torch.allclose(out, torch.full((5, 5), 0.5))


def test_multikernel_weights_method():
    k1 = _ConstantKernel(1.0); k2 = _ConstantKernel(2.0)
    m = MultiKernel([k1, k2])
    weights = m.weights()
    assert abs(weights.sum().item() - 1.0) < 1e-5
    assert weights.shape == (2,)


def test_multihead_grad_flow():
    """Multi-head gradients flow to all heads."""
    import torch.nn as nn
    class _LearnableKernel(Kernel):
        def __init__(self): super().__init__(); self.scale = nn.Parameter(torch.tensor(1.0))
        def forward(self, X1, X2): return self.scale * (X1 @ X2.T)
    m = MultiHead([_LearnableKernel(), _LearnableKernel()])
    X = torch.randn(5, 4)
    out = m(X, X).sum()
    out.backward()
    for h in m.heads:
        assert h.scale.grad is not None
