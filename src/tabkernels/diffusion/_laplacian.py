"""Shared Laplacian utilities for diffusion kernels (Chapter 7 §7.2)."""
from __future__ import annotations

import torch


def degree_matrix(W: torch.Tensor) -> torch.Tensor:
    """Diagonal degree matrix D with D_ii = sum_j W_ij."""
    return W.sum(dim=-1)  # returns the diagonal


def laplacian_sym(W: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Symmetric normalised Laplacian L_sym = I - D^{-1/2} W D^{-1/2}."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got {tuple(W.shape)}")
    d = degree_matrix(W).clamp_min(eps)
    d_inv_sqrt = d.pow(-0.5)
    P = d_inv_sqrt.unsqueeze(-1) * W * d_inv_sqrt.unsqueeze(0)
    N = W.shape[0]
    return torch.eye(N, device=W.device, dtype=W.dtype) - P


def laplacian_comb(W: torch.Tensor) -> torch.Tensor:
    """Combinatorial Laplacian L = D - W."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got {tuple(W.shape)}")
    d = degree_matrix(W)
    return torch.diag(d) - W


def transition_sym(W: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Symmetric random-walk transition P_sym = D^{-1/2} W D^{-1/2}."""
    d = degree_matrix(W).clamp_min(eps)
    d_inv_sqrt = d.pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * W * d_inv_sqrt.unsqueeze(0)
