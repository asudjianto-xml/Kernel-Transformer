"""Tests for tabkernels.diffusion (Chapter 7)."""
import torch
import pytest
from tabkernels.diffusion import (
    HeatKernel, RegLaplacianKernel, PPRKernel, PStepKernel,
    ChebyshevKernel, CommuteTimeKernel, LearnableSpectralKernel,
    laplacian_sym, laplacian_comb, transition_sym,
)


@pytest.fixture
def W_sym():
    """A symmetric non-negative affinity."""
    torch.manual_seed(0)
    M = torch.randn(10, 10).abs() + 0.1
    return (M + M.T) / 2


def test_laplacian_sym_psd(W_sym):
    """L_sym is PSD with smallest eigenvalue 0."""
    L = laplacian_sym(W_sym)
    eigvals = torch.linalg.eigvalsh(L)
    assert eigvals.min() > -1e-5
    # Constant vector should be in the kernel of L_sym (up to D^{1/2} normalisation).
    assert eigvals[0] < 0.01


def test_laplacian_sym_symmetric(W_sym):
    L = laplacian_sym(W_sym)
    assert torch.allclose(L, L.T, atol=1e-5)


def test_heat_kernel_symmetric(W_sym):
    K = HeatKernel(t=0.5)(W_sym)
    assert torch.allclose(K, K.T, atol=1e-5)


def test_heat_kernel_psd(W_sym):
    K = HeatKernel(t=0.5)(W_sym)
    eigvals = torch.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-5


def test_heat_kernel_t_zero_limit():
    """At t -> 0, heat kernel approaches identity."""
    M = torch.randn(8, 8).abs() + 0.1
    W = (M + M.T) / 2
    K = HeatKernel(t=1e-4)(W)
    I = torch.eye(8)
    assert torch.allclose(K, I, atol=1e-3)


def test_reg_laplacian_psd(W_sym):
    K = RegLaplacianKernel(alpha=1.0)(W_sym)
    eigvals = torch.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-5


def test_reg_laplacian_alpha_zero():
    """At alpha = 0, reg-Laplacian is identity."""
    M = torch.randn(6, 6).abs() + 0.1
    W = (M + M.T) / 2
    K = RegLaplacianKernel(alpha=0.0)(W)
    I = torch.eye(6)
    assert torch.allclose(K, I, atol=1e-5)


def test_ppr_closed_form_matches_appnp(W_sym):
    """APPNP at large K approaches the closed-form PPR."""
    closed = PPRKernel(alpha=0.1, n_iter=0)(W_sym)
    appnp = PPRKernel(alpha=0.1, n_iter=200)(W_sym)
    assert torch.allclose(closed, appnp, atol=1e-3)


def test_ppr_alpha_one_limit():
    """At alpha = 1, PPR returns alpha * I."""
    M = torch.randn(6, 6).abs() + 0.1
    W = (M + M.T) / 2
    K = PPRKernel(alpha=0.99)(W)
    I = torch.eye(6)
    # PPR with alpha=0.99 is approximately alpha * I.
    assert torch.allclose(K, 0.99 * I, atol=0.05)


def test_pstep_p_one_matches_transition(W_sym):
    """1-step random walk equals P_sym."""
    K = PStepKernel(p=1)(W_sym)
    P = transition_sym(W_sym)
    assert torch.allclose(K, P, atol=1e-5)


def test_chebyshev_default_is_identity(W_sym):
    """Chebyshev with theta=[1, 0, 0, ...] is identity (T_0 = I)."""
    theta = torch.zeros(5); theta[0] = 1.0
    K = ChebyshevKernel(K=5, theta=theta)(W_sym)
    I = torch.eye(W_sym.shape[0])
    assert torch.allclose(K, I, atol=1e-5)


def test_commute_time_symmetric(W_sym):
    K = CommuteTimeKernel()(W_sym)
    assert torch.allclose(K, K.T, atol=1e-4)


def test_learnable_spectral_symmetric(W_sym):
    K = LearnableSpectralKernel(n_bands=8)(W_sym)
    assert torch.allclose(K, K.T, atol=1e-5)


def test_learnable_spectral_grad_flow(W_sym):
    """Gradient flows to the learnable theta."""
    m = LearnableSpectralKernel(n_bands=8)
    K = m(W_sym)
    loss = K.sum()
    loss.backward()
    assert m.theta.grad is not None
    assert m.theta.grad.abs().sum() > 0
