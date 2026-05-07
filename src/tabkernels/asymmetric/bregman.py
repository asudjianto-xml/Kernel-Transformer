"""Bregman divergence kernel (Chapter 8 §8.3)."""
from __future__ import annotations

from typing import Callable

import torch

from tabkernels.core.base import Kernel


class BregmanKernel(Kernel):
    """Kernel induced by a Bregman divergence: k(x, y) = exp(-D_phi(x, y)).

    Parameters
    ----------
    phi : callable
        Strictly convex function phi: R^d -> R, applied row-wise.
    phi_grad : callable
        Gradient of phi, returning shape (N, d) given input (N, d).
    """

    def __init__(self, phi: Callable[[torch.Tensor], torch.Tensor],
                 phi_grad: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.phi = phi
        self.phi_grad = phi_grad

    def divergence(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Compute D_phi(X1[i], X2[j]) for all i, j."""
        # phi(x) - phi(y) - <grad phi(y), x - y>.
        phi_x = self.phi(X1)  # (N1,)
        phi_y = self.phi(X2)  # (N2,)
        gy = self.phi_grad(X2)  # (N2, d)
        # Term <gy_j, x_i - y_j> = gy_j @ x_i - gy_j @ y_j.
        cross = X1 @ gy.T  # (N1, N2)
        gy_dot_y = (gy * X2).sum(-1)  # (N2,)
        # D[i, j] = phi_x[i] - phi_y[j] - cross[i, j] + gy_dot_y[j].
        return phi_x.unsqueeze(-1) - phi_y.unsqueeze(0) - cross + gy_dot_y.unsqueeze(0)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.divergence(X1, X2))

    def is_symmetric(self) -> bool:
        return False  # generally asymmetric

    def is_psd(self) -> bool:
        return False


def squared_euclidean_kernel() -> BregmanKernel:
    """phi(x) = ||x||^2 / 2 -> D_phi = ||x - y||^2 / 2 (symmetric special case)."""
    return BregmanKernel(
        phi=lambda x: 0.5 * (x ** 2).sum(-1),
        phi_grad=lambda x: x,
    )


def kl_kernel() -> BregmanKernel:
    """phi(x) = sum x_i log x_i (negative entropy on simplex) -> D_phi = KL.

    Note: assumes inputs are on the simplex (positive, rows sum to 1).
    """
    eps = 1e-12

    def _phi(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp_min(eps) * x.clamp_min(eps).log()).sum(-1)

    def _phi_grad(x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(eps).log() + 1.0

    return BregmanKernel(phi=_phi, phi_grad=_phi_grad)
