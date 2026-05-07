"""Unit tests for :mod:`tabkernels.kernels.local`.

Each test is referenced by AGENT-CH04 brief Section 6.
"""
from __future__ import annotations

import pytest
import torch

from tabkernels.kernels.local import (
    CauchyKernel,
    EpanechnikovKernel,
    LearnedBandwidthRBF,
    MahalanobisKernel,
    RBFKernel,
)


# ---------------------------------------------------------------------
# 1. RBF self-kernel
# ---------------------------------------------------------------------
def test_rbf_self_kernel_is_one() -> None:
    """``k(x, x) = 1`` for the global-bandwidth RBF."""
    torch.manual_seed(0)
    X = torch.randn(8, 4)
    k = RBFKernel(bandwidth="global", sigma=1.0)
    G = k(X, X)
    diag = torch.diagonal(G)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)

    # Also true for fixed-numeric bandwidth.
    k2 = RBFKernel(bandwidth=0.7)
    G2 = k2(X, X)
    assert torch.allclose(
        torch.diagonal(G2), torch.ones(X.shape[0]), atol=1e-6
    )


# ---------------------------------------------------------------------
# 2. RBF symmetry
# ---------------------------------------------------------------------
def test_rbf_symmetry() -> None:
    """``k(X1, X2) = k(X2, X1)^T``."""
    torch.manual_seed(1)
    X1 = torch.randn(5, 3)
    X2 = torch.randn(7, 3)
    k = RBFKernel(bandwidth="global", sigma=1.5)
    G12 = k(X1, X2)
    G21 = k(X2, X1)
    assert torch.allclose(G12, G21.T, atol=1e-6)


# ---------------------------------------------------------------------
# 3. RBF PSD
# ---------------------------------------------------------------------
def test_rbf_psd_global() -> None:
    """The Gram matrix of the global-bandwidth RBF is PSD."""
    torch.manual_seed(2)
    X = torch.randn(20, 5)
    k = RBFKernel(bandwidth="global", sigma=1.0)
    G = k(X, X)
    # Symmetrise for numerical safety, then take the smallest eigenvalue.
    G_sym = 0.5 * (G + G.T)
    eigvals = torch.linalg.eigvalsh(G_sym)
    assert eigvals.min().item() >= -1e-5
    assert k.is_psd() is True


# ---------------------------------------------------------------------
# 4. Self-tuning is density-adaptive
# ---------------------------------------------------------------------
def test_rbf_self_tuning_density_adaptive() -> None:
    """On multi-scale data, self-tuning sigma_i varies by point."""
    torch.manual_seed(3)
    # Two clusters at very different scales: one tight, one wide.
    tight = 0.05 * torch.randn(20, 2)
    wide = torch.tensor([5.0, 5.0]) + 1.0 * torch.randn(20, 2)
    X = torch.cat([tight, wide], dim=0)

    k = RBFKernel(bandwidth="self_tuning", K=5)
    sigma = k._self_tuning_sigma(X)
    # The wide cluster should have meaningfully larger sigma_i.
    sigma_tight = sigma[:20].mean().item()
    sigma_wide = sigma[20:].mean().item()
    assert sigma_wide > 5.0 * sigma_tight, (
        f"sigma_wide={sigma_wide} not >> sigma_tight={sigma_tight}"
    )

    # Calling forward on the same X should produce a finite Gram.
    G = k(X, X)
    assert torch.isfinite(G).all()
    # Self-affinity is 1 (D2 = 0 in the exponent).
    assert torch.allclose(
        torch.diagonal(G), torch.ones(X.shape[0]), atol=1e-6
    )
    # is_psd() reports approximate-PSD only:
    assert k.is_psd() is False


# ---------------------------------------------------------------------
# 5. Learned-bandwidth RBF: regulariser prevents collapse
# ---------------------------------------------------------------------
def test_learned_rbf_regularizer_prevents_collapse() -> None:
    """Training with the regulariser keeps sigma bounded away from 0."""
    torch.manual_seed(4)
    X = torch.randn(32, 4)
    y = (X[:, 0] > 0).float()  # toy binary target

    def train_one(reg: float, steps: int = 200) -> torch.Tensor:
        torch.manual_seed(4)
        kernel = LearnedBandwidthRBF(d_in=4, hidden=16, regularizer=reg)
        opt = torch.optim.Adam(kernel.parameters(), lr=5e-2)
        for _ in range(steps):
            opt.zero_grad()
            G = kernel(X, X)
            # NW-like loss that strongly rewards memorisation:
            # match a one-hot target encoding via row-stochastic G.
            row = G / G.sum(dim=1, keepdim=True).clamp_min(1e-8)
            yhat = row @ y
            loss = ((yhat - y) ** 2).mean()
            if reg > 0:
                loss = loss + kernel.regularization_loss(X)
            loss.backward()
            opt.step()
        return kernel._sigma(X).detach()

    sigma_with = train_one(reg=1e-1)
    sigma_without = train_one(reg=0.0)

    # With regulariser sigma stays bounded above 0 and is larger on
    # average than without; without it, sigma can collapse.
    assert sigma_with.min().item() > 1e-2
    assert sigma_with.mean().item() > sigma_without.mean().item()


# ---------------------------------------------------------------------
# 6. Mahalanobis with A = I reduces to RBF
# ---------------------------------------------------------------------
def test_mahalanobis_reduces_to_rbf() -> None:
    """Mahalanobis with L = I matches the unit-bandwidth RBF."""
    torch.manual_seed(5)
    X = torch.randn(10, 3)
    rbf = RBFKernel(bandwidth=1.0)
    mah = MahalanobisKernel(d_in=3)  # L initialised to I
    G_rbf = rbf(X, X)
    G_mah = mah(X, X)
    assert torch.allclose(G_rbf, G_mah, atol=1e-6)
    assert mah.is_psd() is True


# ---------------------------------------------------------------------
# 7. Cauchy PSD at nu = 1, d = 2
# ---------------------------------------------------------------------
def test_cauchy_psd_at_nu_1_dim_2() -> None:
    """At nu = 1 and d = 2, Schoenberg gives PSD."""
    torch.manual_seed(6)
    X = torch.randn(25, 2)
    k = CauchyKernel(nu=1.0)
    G = k(X, X)
    G_sym = 0.5 * (G + G.T)
    eigvals = torch.linalg.eigvalsh(G_sym)
    assert eigvals.min().item() >= -1e-5
    # Self-affinity is 1.
    assert torch.allclose(
        torch.diagonal(G), torch.ones(X.shape[0]), atol=1e-6
    )
    # nu must be positive.
    with pytest.raises(ValueError):
        CauchyKernel(nu=-0.1)
    assert k.is_psd() is True


# ---------------------------------------------------------------------
# 8. Epanechnikov intrinsic sparsity
# ---------------------------------------------------------------------
def test_epanechnikov_intrinsic_sparsity() -> None:
    """At ``h = 0.5`` on standard-normal data, > 50% of pairs are 0."""
    torch.manual_seed(7)
    X = torch.randn(80, 4)
    k = EpanechnikovKernel(bandwidth=0.5)
    G = k(X, X)
    zero_frac = (G == 0).float().mean().item()
    assert zero_frac > 0.5

    # Biweight (power=2) and triweight (power=3) variants smoke test:
    for power in (2, 3):
        k_p = EpanechnikovKernel(bandwidth=0.5, power=power)
        G_p = k_p(X, X)
        assert (G_p >= 0).all()
        assert (G_p <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------
# 9. Epanechnikov compact support
# ---------------------------------------------------------------------
def test_epanechnikov_compact_support() -> None:
    """``k(x, x') = 0`` whenever ``||x - x'|| > h``."""
    h = 1.0
    X1 = torch.tensor([[0.0, 0.0]])
    X2 = torch.tensor([[2.0, 0.0], [0.6, 0.0], [0.0, 1.5]])
    k = EpanechnikovKernel(bandwidth=h)
    G = k(X1, X2)
    # Far points (distance 2.0 and 1.5) should be exactly 0.
    assert G[0, 0].item() == 0.0
    assert G[0, 2].item() == 0.0
    # Near point (distance 0.6) should be > 0.
    assert G[0, 1].item() > 0.0
    # Constructor input validation:
    with pytest.raises(ValueError):
        EpanechnikovKernel(bandwidth=-1.0)
    with pytest.raises(ValueError):
        EpanechnikovKernel(bandwidth=1.0, power=0)
    # Symmetric, PSD flagged True.
    assert k.is_symmetric() is True
    assert k.is_psd() is True


# ---------------------------------------------------------------------
# Extra: exercise non-default code paths for coverage
# ---------------------------------------------------------------------
def test_rbf_invalid_bandwidth_raises() -> None:
    with pytest.raises(ValueError):
        RBFKernel(bandwidth="banana")  # type: ignore[arg-type]


def test_rbf_self_tuning_cross_inputs() -> None:
    """Self-tuning RBF accepts X1 != X2 and produces finite Gram."""
    torch.manual_seed(8)
    X1 = torch.randn(12, 3)
    X2 = torch.randn(9, 3)
    k = RBFKernel(bandwidth="self_tuning", K=4)
    G = k(X1, X2)
    assert G.shape == (12, 9)
    assert torch.isfinite(G).all()
    assert (G >= 0).all() and (G <= 1.0 + 1e-6).all()


def test_learned_rbf_smoke() -> None:
    """LearnedBandwidthRBF runs forward + regularizer in one shot."""
    torch.manual_seed(9)
    X = torch.randn(6, 3)
    k = LearnedBandwidthRBF(d_in=3, hidden=8, regularizer=1e-2)
    G = k(X, X)
    assert G.shape == (6, 6)
    # Self-affinity 1.
    assert torch.allclose(
        torch.diagonal(G), torch.ones(6), atol=1e-6
    )
    reg = k.regularization_loss(X)
    assert reg.item() > 0
    assert k.is_psd() is False
