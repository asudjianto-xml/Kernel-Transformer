"""Sparsemax and $\\alpha$-entmax sparsification (Chapter 5 §5.5).

Sparsemax (:cite:`martins2016sparsemax`): Euclidean projection onto the simplex.
$\\alpha$-entmax (:cite:`peters2019sparse`): one-parameter family interpolating
softmax ($\\alpha = 1$) and sparsemax ($\\alpha = 2$).
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Sparsifier


def sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Closed-form sparsemax along the given dimension.

    Returns the Euclidean projection of ``scores`` onto the probability simplex.
    Output rows sum to 1 and contain exact zeros below the threshold tau(s).
    """
    s_sorted, _ = scores.sort(dim=dim, descending=True)
    cumsum = s_sorted.cumsum(dim=dim)
    rng = torch.arange(1, scores.shape[dim] + 1, device=scores.device,
                       dtype=scores.dtype)
    rng_shape = [1] * scores.ndim
    rng_shape[dim] = -1
    rng = rng.view(*rng_shape)
    # Threshold condition: 1 + i * s_(i) > sum_{j<=i} s_(j)
    cond = 1.0 + rng * s_sorted > cumsum
    k = cond.sum(dim=dim, keepdim=True).to(scores.dtype)
    # cumsum at index k - 1
    sum_k = (s_sorted * cond.to(scores.dtype)).sum(dim=dim, keepdim=True)
    tau = (sum_k - 1.0) / k
    return (scores - tau).clamp_min(0.0)


def entmax_alpha(scores: torch.Tensor, alpha: float = 1.5,
                  dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    """$\\alpha$-entmax via bisection.

    For ``alpha = 1`` this reduces to softmax. For ``alpha = 2`` it
    coincides with :func:`sparsemax`. For intermediate values, a
    bisection search finds the threshold tau such that the output sums to 1.
    """
    if abs(alpha - 1.0) < 1e-6:
        return torch.softmax(scores, dim=dim)
    if abs(alpha - 2.0) < 1e-6:
        return sparsemax(scores, dim=dim)
    # General alpha: ensure alpha > 1.
    if alpha <= 1.0:
        raise ValueError(f"entmax requires alpha > 1, got {alpha}")
    # Bisection on tau s.t. p = ((alpha - 1) (s - tau))_+^(1/(alpha-1)) sums to 1.
    s_max = scores.max(dim=dim, keepdim=True).values
    scores = scores - s_max  # for numerical stability
    lo = scores.min(dim=dim, keepdim=True).values - 1.0
    hi = scores.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        tau = (lo + hi) / 2
        p = ((alpha - 1) * (scores - tau)).clamp_min(0.0).pow(1 / (alpha - 1))
        Z = p.sum(dim=dim, keepdim=True)
        # If Z > 1, tau is too low (raise it); if Z < 1, tau is too high (lower it).
        too_low = Z > 1.0
        lo = torch.where(too_low, tau, lo)
        hi = torch.where(too_low, hi, tau)
    tau = (lo + hi) / 2
    p = ((alpha - 1) * (scores - tau)).clamp_min(0.0).pow(1 / (alpha - 1))
    Z = p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return p / Z


class SparsemaxSparsifier(Sparsifier):
    """Apply sparsemax along the last dimension of a square score matrix.

    Treats the input W as a score matrix; each row is projected onto the
    simplex.  The result has rows that sum to 1 with exact zeros below the
    threshold.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return sparsemax(W, dim=-1)


class EntmaxSparsifier(Sparsifier):
    """Apply $\\alpha$-entmax along the last dimension."""

    def __init__(self, alpha: float = 1.5, n_iter: int = 50):
        super().__init__()
        if alpha <= 1.0:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        self.alpha = alpha
        self.n_iter = n_iter

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return entmax_alpha(W, alpha=self.alpha, dim=-1, n_iter=self.n_iter)
