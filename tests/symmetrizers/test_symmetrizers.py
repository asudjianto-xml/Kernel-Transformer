"""Tests for tabkernels.symmetrizers (Chapter 6)."""
import torch
import pytest
from tabkernels.symmetrizers import (
    AdditiveSymmetrizer, MaxOrSymmetrizer, MutualAndSymmetrizer, SinkhornSymmetrizer,
)


@pytest.fixture
def W_asymmetric():
    """A clearly-asymmetric non-negative matrix."""
    torch.manual_seed(0)
    return torch.randn(8, 8).abs()


@pytest.fixture
def W_knn_pattern():
    """An asymmetric k-NN-style mask."""
    torch.manual_seed(0)
    W = torch.randn(8, 8).abs()
    # Hard k=2 per row.
    _, idx = W.topk(2, dim=-1)
    mask = torch.zeros_like(W)
    rows = torch.arange(8).unsqueeze(1).expand_as(idx)
    mask[rows, idx] = 1.0
    return W * mask


def assert_symmetric(W, atol=1e-6):
    assert torch.allclose(W, W.T, atol=atol), f"Not symmetric: max diff {(W - W.T).abs().max()}"


def test_additive_symmetric(W_asymmetric):
    s = AdditiveSymmetrizer()
    out = s(W_asymmetric)
    assert_symmetric(out)


def test_additive_recovery_of_symmetric():
    """If input is already symmetric, additive returns it unchanged."""
    torch.manual_seed(0)
    M = torch.randn(8, 8); W = (M + M.T) / 2
    out = AdditiveSymmetrizer()(W)
    assert torch.allclose(out, W, atol=1e-6)


def test_additive_psd_preserving():
    """If input is PSD, additive symmetric is PSD (in fact equal to input if input was symmetric)."""
    torch.manual_seed(0)
    A = torch.randn(8, 4); W = A @ A.T  # PSD
    out = AdditiveSymmetrizer()(W)
    eigvals = torch.linalg.eigvalsh(out)
    assert eigvals.min() > -1e-5


def test_max_or_symmetric(W_asymmetric):
    s = MaxOrSymmetrizer()
    out = s(W_asymmetric)
    assert_symmetric(out)


def test_max_or_unifies_supports(W_knn_pattern):
    """OR-symmetrisation has support = union of (W, W^T) supports."""
    s = MaxOrSymmetrizer()
    out = s(W_knn_pattern)
    union_support = ((W_knn_pattern > 0) | (W_knn_pattern.T > 0)).float()
    out_support = (out > 0).float()
    assert torch.allclose(out_support, union_support)


def test_mutual_and_symmetric(W_knn_pattern):
    s = MutualAndSymmetrizer()
    out = s(W_knn_pattern)
    assert_symmetric(out)


def test_mutual_and_intersects_supports(W_knn_pattern):
    """AND-symmetrisation has support = intersection of (W, W^T) supports."""
    s = MutualAndSymmetrizer()
    out = s(W_knn_pattern)
    intersect_support = ((W_knn_pattern > 0) & (W_knn_pattern.T > 0)).float()
    out_support = (out > 0).float()
    assert torch.allclose(out_support, intersect_support)


def test_mutual_and_sparser_than_or(W_knn_pattern):
    """AND-symmetrisation is at least as sparse as OR-symmetrisation."""
    n_or = (MaxOrSymmetrizer()(W_knn_pattern) > 0).sum().item()
    n_and = (MutualAndSymmetrizer()(W_knn_pattern) > 0).sum().item()
    assert n_and <= n_or


def test_sinkhorn_symmetric(W_asymmetric):
    s = SinkhornSymmetrizer(iters=30)
    out = s(W_asymmetric)
    assert_symmetric(out)


def test_sinkhorn_approximate_marginal():
    """Sinkhorn output has approximately uniform row sums."""
    torch.manual_seed(0)
    W = torch.randn(10, 10).abs() + 0.1
    s = SinkhornSymmetrizer(iters=50)
    out = s(W)
    row_sums = out.sum(dim=-1)
    target = torch.full_like(row_sums, 1.0)
    # The final additive symmetrisation may shift sums slightly; tolerance is generous.
    assert (row_sums - target).abs().max() < 0.5


def test_sinkhorn_invalid_iters():
    with pytest.raises(ValueError):
        SinkhornSymmetrizer(iters=0)


def test_all_symmetrizers_handle_already_symmetric():
    torch.manual_seed(0)
    M = torch.randn(8, 8).abs()
    W = (M + M.T) / 2  # symmetric
    for s in [AdditiveSymmetrizer(), MaxOrSymmetrizer(), MutualAndSymmetrizer()]:
        out = s(W)
        assert_symmetric(out)
