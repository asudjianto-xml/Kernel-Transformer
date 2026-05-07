"""Sinkhorn soft top-$k$ sparsification (Chapter 5 §5.4).

Differentiable top-$k$ via entropy-regularised optimal transport, due to
:cite:`xie2020softtopk` and :cite:`cuturi2013sinkhorn`.
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Sparsifier


class SinkhornTopKSparsifier(Sparsifier):
    """Differentiable top-$k$ via Sinkhorn iteration.

    Each row of W is "transported" onto a target marginal (k/N, ..., k/N) via
    entropy-regularised Sinkhorn iterations, producing a soft top-$k$ mask
    M in [0, 1]^{N x N} with approximately k entries per row. The output is
    the element-wise product M * W.

    Parameters
    ----------
    k : int
        Approximate number of entries to keep per row.
    iters : int, default 20
        Number of Sinkhorn iterations.
    eps : float, default 0.1
        Entropy-regularisation temperature. Smaller -> sharper top-k.
    """

    def __init__(self, k: int, iters: int = 20, eps: float = 0.1):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        self.k = k
        self.iters = iters
        self.eps = eps

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square (N, N), got {tuple(W.shape)}")
        N = W.shape[0]
        if self.k > N:
            raise ValueError(f"k={self.k} > N={N}")

        # Work in log-space for stability.
        log_M = W / self.eps
        target = torch.full((N,), self.k / N, device=W.device, dtype=W.dtype).log()
        src = torch.zeros(N, device=W.device, dtype=W.dtype)
        for _ in range(self.iters):
            # Row normalisation -> sum to k/N
            log_M = log_M - log_M.logsumexp(dim=-1, keepdim=True) + target.unsqueeze(0)
            # Column normalisation -> sum to k/N
            log_M = log_M - log_M.logsumexp(dim=0, keepdim=True) + target.unsqueeze(-1)
        return log_M.exp() * W
