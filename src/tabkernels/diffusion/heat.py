"""Heat kernel diffusion (Chapter 7 §7.3).

K_t = exp(-t L_sym) computed via eigendecomposition (small N) or Chebyshev
polynomial expansion (any N).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tabkernels.core.base import Kernel
from tabkernels.diffusion._laplacian import laplacian_sym


class HeatKernel(Kernel):
    """Heat kernel diffusion: K_t = exp(-t L_sym).

    The kernel takes an affinity matrix W as input via ``forward(W, _)``.
    The second argument is unused (since the diffusion kernel is N x N
    over the same nodes).

    Parameters
    ----------
    t : float
        Diffusion time.
    """

    def __init__(self, t: float = 1.0):
        super().__init__()
        if t <= 0:
            raise ValueError("t must be > 0")
        self.t = t

    def forward(self, W: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        L = laplacian_sym(W)
        # Eigendecomposition is fine for tabular-scale N. For larger graphs,
        # see the Chebyshev approximation in chebyshev.py.
        eigvals, eigvecs = torch.linalg.eigh(L)
        return eigvecs @ torch.diag(torch.exp(-self.t * eigvals)) @ eigvecs.T

    def is_psd(self) -> bool:
        return True
