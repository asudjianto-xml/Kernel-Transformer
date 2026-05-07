"""Chebyshev polynomial spectral filter (Chapter 7 §7.7).

K = sum_{k=0}^{K-1} theta_k T_k(L_tilde) where L_tilde is the rescaled
Laplacian with eigenvalues in [-1, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tabkernels.core.base import Kernel
from tabkernels.diffusion._laplacian import laplacian_sym


class ChebyshevKernel(Kernel):
    """Chebyshev polynomial filter on the Laplacian.

    Parameters
    ----------
    K : int
        Polynomial degree.
    theta : torch.Tensor or None
        Coefficients (K,). If None, learned via nn.Parameter; if provided,
        used as fixed coefficients.
    """

    def __init__(self, K: int = 10, theta: torch.Tensor | None = None):
        super().__init__()
        if K < 1:
            raise ValueError("K must be >= 1")
        self.K = K
        if theta is None:
            self.theta = nn.Parameter(torch.zeros(K))
            with torch.no_grad():
                self.theta[0] = 1.0  # default initialisation: identity-ish filter
        else:
            theta = torch.as_tensor(theta, dtype=torch.float32)
            if theta.shape != (K,):
                raise ValueError(f"theta must have shape ({K},), got {tuple(theta.shape)}")
            self.register_buffer("theta", theta)

    def forward(self, W: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        L = laplacian_sym(W)
        # Rescale to [-1, 1]: L_tilde = (2/lambda_max) L - I.
        lambda_max = torch.linalg.eigvalsh(L)[-1].clamp_min(1.0)
        N = W.shape[0]
        I = torch.eye(N, device=W.device, dtype=W.dtype)
        L_tilde = (2.0 / lambda_max) * L - I

        # Chebyshev recurrence: T_0 = I, T_1 = L_tilde, T_{k+1} = 2 L_tilde T_k - T_{k-1}.
        T_prev = I
        T_curr = L_tilde
        out = self.theta[0] * T_prev
        if self.K >= 2:
            out = out + self.theta[1] * T_curr
        for k in range(2, self.K):
            T_next = 2 * L_tilde @ T_curr - T_prev
            out = out + self.theta[k] * T_next
            T_prev, T_curr = T_curr, T_next
        return out
