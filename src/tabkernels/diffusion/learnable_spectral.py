"""Learnable spectral filter (Chapter 7 §7.9).

K = sum_i f_theta(lambda_i) u_i u_i^T for a learnable spectral filter f_theta.

This implementation uses a piecewise-linear filter over the eigenvalue range
parameterised by `n_bands` learnable knots. Cost is O(N^3) for the
eigendecomposition; intended for small-N tabular settings.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tabkernels.core.base import Kernel
from tabkernels.diffusion._laplacian import laplacian_sym


class LearnableSpectralKernel(Kernel):
    """Piecewise-linear learnable spectral filter on L_sym.

    Parameters
    ----------
    n_bands : int
        Number of knots in the piecewise-linear filter; spans [0, 2].
    """

    def __init__(self, n_bands: int = 8):
        super().__init__()
        if n_bands < 2:
            raise ValueError("n_bands must be >= 2")
        self.n_bands = n_bands
        # Knots evenly spaced over [0, 2] (the eigenvalue range of L_sym).
        self.register_buffer("knots", torch.linspace(0, 2, n_bands))
        self.theta = nn.Parameter(torch.ones(n_bands))

    def _filter(self, lam: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear interpolation: f(lambda) interpolates theta at knots."""
        # Find the right segment for each eigenvalue.
        knots = self.knots
        # Indices of knots immediately above each lam.
        idx = torch.bucketize(lam, knots).clamp(1, self.n_bands - 1)
        # Linear interpolation between (knots[idx-1], theta[idx-1]) and (knots[idx], theta[idx]).
        x0 = knots[idx - 1]; x1 = knots[idx]
        y0 = self.theta[idx - 1]; y1 = self.theta[idx]
        t = (lam - x0) / (x1 - x0).clamp_min(1e-12)
        return y0 + t * (y1 - y0)

    def forward(self, W: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        L = laplacian_sym(W)
        eigvals, eigvecs = torch.linalg.eigh(L)
        f_vals = self._filter(eigvals.clamp(0, 2))
        return eigvecs @ torch.diag(f_vals) @ eigvecs.T
