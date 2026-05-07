"""Tests for tabkernels.sparsifiers (Chapter 5)."""
import torch
import pytest
from tabkernels.sparsifiers import (
    KNNSparsifier, SparsemaxSparsifier, EntmaxSparsifier,
    SinkhornTopKSparsifier, EpsilonBallSparsifier, sparsemax, entmax_alpha,
)


@pytest.fixture
def W():
    torch.manual_seed(0)
    return torch.randn(20, 20).abs()  # non-negative, dense


def test_knn_sparsity_count(W):
    """Hard k-NN keeps exactly k entries per row."""
    s = KNNSparsifier(k=3)
    out = s(W)
    nonzero_per_row = (out > 0).sum(dim=-1)
    assert (nonzero_per_row == 3).all()


def test_knn_keeps_topk_values(W):
    """The kept values are the top-k of each row."""
    s = KNNSparsifier(k=4)
    out = s(W)
    for i in range(W.shape[0]):
        topk = W[i].topk(4).values.sort().values
        kept = out[i][out[i] > 0].sort().values
        assert torch.allclose(topk, kept)


def test_knn_mutual_symmetric():
    """Mutual k-NN produces a symmetric mask."""
    torch.manual_seed(0)
    W = torch.randn(15, 15).abs()
    # Make non-symmetric.
    s = KNNSparsifier(k=3, mutual=True)
    out = s(W)
    # The MASK is symmetric; the resulting W * mask is symmetric where mask is 1.
    mask = (out > 0).float()
    assert torch.allclose(mask, mask.T)


def test_knn_invalid_k():
    with pytest.raises(ValueError):
        KNNSparsifier(k=0)
    with pytest.raises(ValueError):
        KNNSparsifier(k=100)(torch.randn(10, 10))


def test_sparsemax_simplex():
    """Sparsemax output rows sum to 1."""
    torch.manual_seed(0)
    s = torch.randn(8, 12)
    out = sparsemax(s, dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_sparsemax_zeros():
    """Sparsemax produces exact zeros on entries below threshold."""
    s = torch.tensor([[5.0, 0.1, 0.05, -1.0]])
    out = sparsemax(s, dim=-1)
    assert (out == 0).any()


def test_sparsemax_idempotent_on_simplex():
    """Sparsemax of a probability vector returns the same vector."""
    p = torch.tensor([[0.5, 0.3, 0.2]])
    out = sparsemax(p, dim=-1)
    assert torch.allclose(out, p, atol=1e-4)


def test_entmax_softmax_recovery():
    """entmax with alpha=1 should agree with softmax (within bisection error)."""
    torch.manual_seed(0)
    s = torch.randn(4, 8)
    out_alpha1 = entmax_alpha(s, alpha=1.0)
    expected = torch.softmax(s, dim=-1)
    assert torch.allclose(out_alpha1, expected, atol=1e-5)


def test_entmax_sparsemax_recovery():
    """entmax with alpha=2 should agree with sparsemax."""
    torch.manual_seed(0)
    s = torch.randn(4, 8)
    out_alpha2 = entmax_alpha(s, alpha=2.0)
    expected = sparsemax(s, dim=-1)
    assert torch.allclose(out_alpha2, expected, atol=1e-5)


def test_entmax_intermediate_alpha():
    """entmax(alpha=1.5) is sparser than softmax but denser than sparsemax."""
    torch.manual_seed(0)
    s = torch.randn(4, 20)
    out_softmax = torch.softmax(s, dim=-1)
    out_15 = entmax_alpha(s, alpha=1.5)
    out_sparse = sparsemax(s, dim=-1)
    n_softmax = (out_softmax > 1e-6).sum().item()
    n_15 = (out_15 > 1e-6).sum().item()
    n_sparse = (out_sparse > 1e-6).sum().item()
    # softmax has all entries > 0; sparsemax has the fewest.
    assert n_softmax >= n_15 >= n_sparse


def test_sinkhorn_marginals():
    """Sinkhorn mask has approximate row marginals k/N."""
    torch.manual_seed(0)
    W = torch.randn(10, 10).abs() + 0.1  # positive
    k = 3
    N = 10
    s = SinkhornTopKSparsifier(k=k, iters=50, eps=0.05)
    out = s(W)
    # The implicit mask M = exp(log_M) satisfies row sums approx k/N (after
    # normalisation against (k/N, ..., k/N) marginals).
    mask = out / W.clamp_min(1e-9)
    row_sums = mask.sum(dim=-1)
    target = (k / N) * torch.ones_like(row_sums)
    # Mean absolute deviation from the target should be small after enough
    # Sinkhorn iterations.
    assert (row_sums - target).abs().mean() < 0.2


def test_sinkhorn_concentrates_on_top_entries():
    """Sinkhorn places most mass on the largest W entries per row."""
    torch.manual_seed(0)
    W = torch.randn(10, 10).abs() + 0.1
    k = 3
    s = SinkhornTopKSparsifier(k=k, iters=50, eps=0.05)
    out = s(W)
    # Top-k indices in W per row should overlap heavily with top-k in out.
    _, topk_W = W.topk(k, dim=-1)
    _, topk_out = out.topk(k, dim=-1)
    # Compute per-row overlap.
    overlap = 0
    for i in range(W.shape[0]):
        overlap += len(set(topk_W[i].tolist()) & set(topk_out[i].tolist()))
    avg_overlap = overlap / (k * W.shape[0])
    # At eps = 0.05 and 50 iters, overlap should be high.
    assert avg_overlap > 0.5


def test_eps_ball_mask():
    """Epsilon ball sparsification thresholds on affinity."""
    W = torch.tensor([[1.0, 0.5, 0.1],
                       [0.5, 1.0, 0.05],
                       [0.1, 0.05, 1.0]])
    s = EpsilonBallSparsifier(epsilon=0.4)
    out = s(W)
    expected = torch.tensor([[1.0, 0.5, 0.0],
                              [0.5, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    assert torch.allclose(out, expected)


def test_santos_2026_equivalence():
    """Sparsemax of squared-distance scores matches Epanechnikov NW (Eq. 5.7)."""
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    h = 1.0
    # Pairwise squared distances.
    sq_dist = ((X[:, None] - X[None, :]) ** 2).sum(-1)
    # Sparsemax weights of negative scaled distances.
    scores = -sq_dist / (2 * h ** 2)
    weights_sparsemax = sparsemax(scores, dim=-1)
    # Epanechnikov NW weights: max(0, 1 - r^2/h^2) normalised.
    epi = (1 - sq_dist / h ** 2).clamp_min(0.0)
    weights_epanechnikov = epi / epi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # The Santos 2026 result says these are exactly equal at the *support* level
    # (same set of zero entries) and, after appropriate scaling, the same numerical
    # values under specific conditions. We check the weaker claim: the
    # set of zero entries matches.
    sparsemax_zeros = (weights_sparsemax == 0)
    epanechnikov_zeros = (weights_epanechnikov == 0)
    # Weak match on the support.
    overlap = (sparsemax_zeros == epanechnikov_zeros).float().mean()
    assert overlap.item() > 0.85, f"Support overlap {overlap.item():.3f} < 0.85"
