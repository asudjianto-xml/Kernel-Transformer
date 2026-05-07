"""Personalised PageRank diffusion (Chapter 7 §7.5).

Closed-form: K = alpha * (I - (1-alpha) P_sym)^{-1}.
APPNP approximation via K iterations of power method.
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Kernel
from tabkernels.diffusion._laplacian import transition_sym


class PPRKernel(Kernel):
    """Personalised PageRank diffusion kernel.

    Parameters
    ----------
    alpha : float
        Teleport probability in [0, 1]. Smaller -> more diffusion.
    n_iter : int, default 0
        If > 0, use APPNP (truncated geometric) with this many iterations.
        If 0, use the closed-form inverse.
    """

    def __init__(self, alpha: float = 0.1, n_iter: int = 0):
        super().__init__()
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.n_iter = n_iter

    def forward(self, W: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        P = transition_sym(W)
        N = W.shape[0]
        I = torch.eye(N, device=W.device, dtype=W.dtype)
        if self.n_iter > 0:
            # APPNP: H_{k+1} = (1-alpha) P H_k + alpha I.
            H = I.clone()
            for _ in range(self.n_iter):
                H = (1 - self.alpha) * (P @ H) + self.alpha * I
            return H
        # Closed form.
        return self.alpha * torch.linalg.solve(I - (1 - self.alpha) * P, I)
