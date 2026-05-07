"""Hard $k$-NN top-$k$ and mutual $k$-NN sparsification (Chapter 5 §5.2-§5.3)."""
from __future__ import annotations

import torch

from tabkernels.core.base import Sparsifier


class KNNSparsifier(Sparsifier):
    """Keep the ``k`` largest entries per row of W; zero the rest.

    If ``mutual=True``, additionally require the entry to be among the top-k
    in the *transpose* (mutual k-NN, AND-symmetrisation).

    Parameters
    ----------
    k : int
        Number of neighbours to keep per row.
    mutual : bool, default False
        If True, only keep edges (i, j) where j is in topk(W_i) AND
        i is in topk(W_j).
    exclude_self : bool, default False
        If True, exclude self-edges (diagonal) from the topk.
    """

    def __init__(self, k: int, mutual: bool = False, exclude_self: bool = False):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.mutual = mutual
        self.exclude_self = exclude_self

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square (N, N), got {tuple(W.shape)}")
        N = W.shape[0]
        if self.k > N:
            raise ValueError(f"k={self.k} > N={N}")

        if self.exclude_self:
            mask_diag = torch.eye(N, device=W.device, dtype=torch.bool)
            W_for_topk = W.masked_fill(mask_diag, float("-inf"))
        else:
            W_for_topk = W

        # Top-k per row.
        _, idx = W_for_topk.topk(self.k, dim=-1, largest=True)
        mask = torch.zeros_like(W)
        rows = torch.arange(N, device=W.device).unsqueeze(1).expand_as(idx)
        mask[rows, idx] = 1.0

        if self.mutual:
            mask = mask * mask.T  # AND-symmetrise

        return W * mask

    def is_differentiable(self) -> bool:
        return False  # index selection is non-differentiable
