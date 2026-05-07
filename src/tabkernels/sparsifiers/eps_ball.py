"""$\\varepsilon$-neighbourhood sparsification (Chapter 5 §5.6).

Keep entries W_ij where the corresponding distance is below a radius
$\\varepsilon$.  Used in Laplacian eigenmaps and other spectral methods.
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Sparsifier


class EpsilonBallSparsifier(Sparsifier):
    """Keep entries within radius ``epsilon`` according to a distance matrix.

    Two usage modes:
      - If a separate distance matrix is provided to ``forward(W, D)``, it
        is used directly.
      - Otherwise the sparsifier interprets ``-W`` as a (squared) distance
        proxy: high affinity ↔ low distance.  The threshold is then on the
        affinity itself.

    Parameters
    ----------
    epsilon : float
        Distance threshold (or affinity threshold in the implicit mode).
    on_distance : bool, default False
        If True, expect ``forward(W, D)`` with a separate distance matrix
        and apply the threshold on D.  Otherwise apply ``W >= epsilon`` as
        the implicit affinity threshold.
    """

    def __init__(self, epsilon: float, on_distance: bool = False):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        self.epsilon = epsilon
        self.on_distance = on_distance

    def forward(self, W: torch.Tensor,
                D: torch.Tensor | None = None) -> torch.Tensor:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square, got {tuple(W.shape)}")
        if self.on_distance:
            if D is None:
                raise ValueError("on_distance=True requires D matrix")
            mask = (D <= self.epsilon).to(W.dtype)
        else:
            mask = (W >= self.epsilon).to(W.dtype)
        return W * mask

    def is_differentiable(self) -> bool:
        return False  # threshold is non-differentiable
