"""Tests for tabkernels.asymmetric (Chapter 8)."""
import torch
import pytest
from tabkernels.asymmetric import (
    BilinearQKKernel, BregmanKernel, squared_euclidean_kernel, kl_kernel,
    DirectedLaplacianKernel, stationary_distribution,
)


def test_bilinear_qk_inner():
    """Inner-product mode produces asymmetric scores in general."""
    torch.manual_seed(0)
    X = torch.randn(8, 4)
    k = BilinearQKKernel(d_in=4, d_h=4, mode="inner")
    K = k(X, X)
    # Generally non-symmetric.
    assert (K - K.T).abs().max() > 1e-3


def test_bilinear_qk_gaussian_diagonal_one():
    """Gaussian mode k(x, x) = exp(-||W_Q x - W_K x||^2). Generally < 1."""
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    k = BilinearQKKernel(d_in=4, d_h=4, mode="gaussian")
    K = k(X, X)
    # Diagonal entries are exp(-||(W_Q - W_K) x_i||^2), which are <= 1.
    assert (torch.diag(K) <= 1.0 + 1e-5).all()


def test_bilinear_qk_psd_check():
    k = BilinearQKKernel(d_in=4, d_h=4)
    assert k.is_psd() is False
    assert k.is_symmetric() is False


def test_bregman_squared_euclidean_is_symmetric():
    """Squared Euclidean is the symmetric special case of Bregman."""
    torch.manual_seed(0)
    X = torch.randn(8, 3)
    k = squared_euclidean_kernel()
    D = k.divergence(X, X)
    assert torch.allclose(D, D.T, atol=1e-5)
    # Diagonal of D is zero.
    assert torch.allclose(torch.diag(D), torch.zeros(8), atol=1e-5)


def test_bregman_kl_asymmetric():
    """KL divergence is asymmetric on the simplex."""
    torch.manual_seed(0)
    # Random simplex vectors.
    X = torch.softmax(torch.randn(6, 4), dim=-1)
    k = kl_kernel()
    D = k.divergence(X, X)
    # Diagonal is zero (KL(x || x) = 0).
    assert torch.allclose(torch.diag(D), torch.zeros(6), atol=1e-4)
    # Off-diagonal generally asymmetric.
    assert (D - D.T).abs().max() > 1e-3


def test_bregman_kernel_non_negative():
    """exp(-D_phi) is non-negative."""
    X = torch.softmax(torch.randn(5, 4), dim=-1)
    k = kl_kernel()
    K = k(X, X)
    assert (K >= 0).all()


def test_stationary_distribution_sums_to_one():
    """Stationary distribution sums to 1."""
    torch.manual_seed(0)
    W = torch.randn(8, 8).abs() + 0.1
    d = W.sum(dim=-1)
    P = W / d.unsqueeze(-1)
    pi = stationary_distribution(P)
    assert abs(pi.sum().item() - 1.0) < 1e-5
    assert (pi >= 0).all()


def test_directed_laplacian_symmetric():
    """The directed Laplacian L_dir is symmetric by construction."""
    torch.manual_seed(0)
    W = torch.randn(8, 8).abs() + 0.1  # asymmetric
    L = DirectedLaplacianKernel()(W)
    assert torch.allclose(L, L.T, atol=1e-5)


def test_directed_laplacian_psd():
    """L_dir is PSD."""
    torch.manual_seed(0)
    W = torch.randn(8, 8).abs() + 0.1
    L = DirectedLaplacianKernel()(W)
    eigvals = torch.linalg.eigvalsh(L)
    assert eigvals.min() > -1e-4


def test_directed_laplacian_symmetric_input_gives_standard_laplacian():
    """When W is symmetric, the directed Laplacian reduces to (or near) the symmetric one."""
    torch.manual_seed(0)
    M = torch.randn(8, 8).abs() + 0.1
    W = (M + M.T) / 2
    L = DirectedLaplacianKernel()(W)
    # The directed Laplacian on symmetric input has eigenvalues in [0, 2].
    eigvals = torch.linalg.eigvalsh(L)
    assert eigvals.min() > -1e-4
    assert eigvals.max() < 2.5  # tolerance for numerical
