"""Directed graph Laplacian (Chapter 8 §8.4).

For an asymmetric W, builds the symmetric Laplacian via the random-walk
stationary distribution.  Reduces an asymmetric graph problem to a symmetric
spectral problem.
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Kernel


def stationary_distribution(P: torch.Tensor, n_iter: int = 100,
                             tol: float = 1e-8) -> torch.Tensor:
    """Compute the stationary distribution of the row-stochastic P via power iteration."""
    N = P.shape[0]
    pi = torch.full((N,), 1.0 / N, device=P.device, dtype=P.dtype)
    for _ in range(n_iter):
        pi_new = pi @ P  # P is row-stochastic (rows sum to 1) so pi @ P sums to 1
        pi_new = pi_new / pi_new.sum().clamp_min(1e-12)
        if (pi_new - pi).abs().max() < tol:
            break
        pi = pi_new
    return pi


class DirectedLaplacianKernel(Kernel):
    """Symmetric Laplacian on a directed graph (Chung 2005).

    L_dir = I - 1/2 (Pi^{1/2} P Pi^{-1/2} + Pi^{-1/2} P^T Pi^{1/2})

    where P = D^{-1} W is the row-stochastic transition matrix and Pi is
    the diagonal of the stationary distribution.
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, W: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square, got {tuple(W.shape)}")
        N = W.shape[0]
        # Row-stochastic P.
        d = W.sum(dim=-1).clamp_min(self.eps)
        P = W / d.unsqueeze(-1)
        # Stationary distribution.
        pi = stationary_distribution(P)
        pi_sqrt = pi.clamp_min(self.eps).sqrt()
        pi_inv_sqrt = 1.0 / pi_sqrt
        # Symmetric Laplacian.
        sym = 0.5 * (
            pi_sqrt.unsqueeze(-1) * P * pi_inv_sqrt.unsqueeze(0)
            + pi_inv_sqrt.unsqueeze(-1) * P.T * pi_sqrt.unsqueeze(0)
        )
        I = torch.eye(N, device=W.device, dtype=W.dtype)
        return I - sym
