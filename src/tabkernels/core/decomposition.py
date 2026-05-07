"""Symmetric / skew-symmetric decomposition primitives.

These are used everywhere — by AttentionBlock subclasses, by audits, by
Ch 13 transparency tooling.
"""
from __future__ import annotations

import torch


def decompose(B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose B into (B_S, B_A) where B_S = (B + B^T)/2 and B_A = (B - B^T)/2."""
    B_T = B.transpose(-2, -1)
    return (B + B_T) / 2, (B - B_T) / 2


def energy_split(B: torch.Tensor) -> tuple[float, float]:
    """Return Frobenius energy fractions (alpha_S, alpha_A) of B."""
    B_S, B_A = decompose(B)
    s = (B_S ** 2).sum().item()
    a = (B_A ** 2).sum().item()
    total = s + a + 1e-12
    return s / total, a / total


def cosine_similarity_F(A: torch.Tensor, B: torch.Tensor) -> float:
    """Frobenius cosine similarity between two matrices."""
    a = A.flatten()
    b = B.flatten()
    denom = (a.norm() * b.norm() + 1e-12)
    return (a @ b / denom).item()


def project_score(S: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply post-hoc decomposition projection to an attention score matrix.

    Args:
        S: (..., N, N) attention scores (pre-softmax).
        mode: one of 'full', 'sym', 'asym'.

    Returns:
        S unchanged ('full'), or (S + S^T)/2 ('sym'), or (S - S^T)/2 ('asym').
    """
    if mode == "full":
        return S
    S_T = S.transpose(-2, -1)
    if mode == "sym":
        return (S + S_T) / 2
    if mode == "asym":
        return (S - S_T) / 2
    raise ValueError(f"unknown mode: {mode!r}")
