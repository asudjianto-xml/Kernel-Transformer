"""Family 1: Local kernels on learned embeddings.

See Chapter 4 of *Kernels and Transformers for Tabular Data*.

This module implements the five Family-1 kernel forms catalogued in
``affinity/kernel_catalog.tex`` Sections sec:gauss-global through
sec:compact:

* :class:`RBFKernel` -- Gaussian RBF with three bandwidth modes
  (global, self-tuning per-point, fixed scalar).
* :class:`LearnedBandwidthRBF` -- per-point learned bandwidth via a
  small MLP with a regulariser to prevent ``sigma -> 0`` collapse.
* :class:`MahalanobisKernel` -- anisotropic Gaussian with a learned
  PSD metric ``A = L L^T``.
* :class:`CauchyKernel` -- Student-t / Cauchy heavy-tailed kernel.
* :class:`EpanechnikovKernel` -- compact-support polynomial kernel
  (with biweight / triweight via the ``power`` argument).

All kernels return ``(N1, N2)`` Gram matrices for inputs ``(X1, X2)``
of shapes ``(N1, d)`` and ``(N2, d)``.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import Kernel
from tabkernels.core.types import FeatureTensor, GramTensor

BandwidthMode = Literal["global", "self_tuning"] | float | int


def _pairwise_sq_dist(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances ``||x_i - y_j||^2`` between rows.

    Wrapper around :func:`torch.cdist` that clamps small negatives that
    can appear from the fused ``a^2 + b^2 - 2 a.b`` formula.
    """
    D = torch.cdist(X1, X2)
    return D.pow(2).clamp_min(0.0)


class RBFKernel(Kernel):
    """Gaussian radial basis function kernel.

    Variants:
        ``bandwidth='global'``: single learnable scalar ``sigma``.
        ``bandwidth='self_tuning'``: per-point ``sigma_i`` set to the
            distance from ``x_i`` to its ``K``-th nearest neighbour
            (Zelnik-Manor & Perona 2004).
        ``bandwidth=<float>``: fixed numeric ``sigma``, not trainable.

    Args:
        bandwidth: see above.
        sigma: initial sigma (only used when ``bandwidth='global'``).
        K: number of neighbours for self-tuning.

    Properties:
        ``is_symmetric() -> True`` for all variants.
        ``is_psd() -> True`` for global and numeric bandwidth;
        approximately True for self-tuning (Schoenberg's theorem
        applies in transformed space, but the per-point scaling
        breaks the strict guarantee).
    """

    def __init__(
        self,
        bandwidth: BandwidthMode = "global",
        sigma: float = 1.0,
        K: int = 7,
    ) -> None:
        super().__init__()
        mode: str
        if isinstance(bandwidth, str):
            if bandwidth not in ("global", "self_tuning"):
                raise ValueError(f"unknown bandwidth: {bandwidth!r}")
            mode = bandwidth
        elif isinstance(bandwidth, (int, float)):
            mode = "fixed"
        else:
            raise ValueError(f"unknown bandwidth: {bandwidth!r}")
        self.mode = mode

        if mode == "global":
            self.log_sigma = nn.Parameter(torch.tensor(float(sigma)).log())
        elif mode == "fixed":
            self.register_buffer(
                "sigma_buf", torch.tensor(float(bandwidth))
            )
        else:  # self_tuning
            self.K = int(K)

    def _sigma_global(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp_min(1e-6)

    def _sigma_fixed(self) -> torch.Tensor:
        sigma_buf: torch.Tensor = self.sigma_buf  # type: ignore[assignment]
        return sigma_buf.clamp_min(1e-6)

    def _self_tuning_sigma(self, X: torch.Tensor) -> torch.Tensor:
        """Per-point bandwidth = distance to K-th NN.

        Uses ``torch.topk(..., k=K+1, largest=False)`` so that the
        nearest neighbour (``self``) is excluded automatically.
        """
        N = X.shape[0]
        D = torch.cdist(X, X)
        k = min(self.K + 1, N)
        kth_dist, _ = D.topk(k, largest=False)
        return kth_dist[:, -1].clamp_min(1e-6)

    def forward(
        self, X1: FeatureTensor, X2: FeatureTensor
    ) -> GramTensor:
        D2 = _pairwise_sq_dist(X1, X2)
        if self.mode == "global":
            sigma = self._sigma_global()
            return torch.exp(-D2 / (sigma * sigma))
        if self.mode == "fixed":
            sigma = self._sigma_fixed()
            return torch.exp(-D2 / (sigma * sigma))
        # self-tuning: sigma_i from X1, sigma_j from X2
        sigma1 = self._self_tuning_sigma(X1)
        if X2.data_ptr() == X1.data_ptr() and X2.shape == X1.shape:
            sigma2 = sigma1
        else:
            sigma2 = self._self_tuning_sigma(X2)
        sigma_ij = sigma1.unsqueeze(1) * sigma2.unsqueeze(0)
        return torch.exp(-D2 / sigma_ij.clamp_min(1e-12))

    def is_psd(self) -> bool:
        return self.mode in ("global", "fixed")


class LearnedBandwidthRBF(Kernel):
    """Per-point learned bandwidth via a small MLP.

    ``sigma_i = softplus(g_phi(x_i))``.  Combined with the symmetric
    Zelnik-Manor product ``sigma_i sigma_j`` in the exponent, this
    gives a fully end-to-end trainable density-adaptive RBF.

    The :meth:`regularization_loss` term should be added to the
    training loss; without it the encoder can drive ``sigma_i -> 0``
    on training points, producing a near-binary affinity that
    memorises the training set.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int = 32,
        regularizer: float = 1e-3,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.regularizer = float(regularizer)

    def _sigma(self, X: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.encoder(X)).squeeze(-1).clamp_min(1e-6)

    def forward(
        self, X1: FeatureTensor, X2: FeatureTensor
    ) -> GramTensor:
        sigma1 = self._sigma(X1)
        sigma2 = self._sigma(X2)
        D2 = _pairwise_sq_dist(X1, X2)
        sigma_ij = sigma1.unsqueeze(1) * sigma2.unsqueeze(0)
        return torch.exp(-D2 / sigma_ij.clamp_min(1e-12))

    def regularization_loss(self, X: FeatureTensor) -> torch.Tensor:
        """Penalise ``sigma -> 0`` collapse.

        Returns ``regularizer * mean(1 / sigma_i^2)``; should be added
        to the training loss before the backward pass.
        """
        sigma = self._sigma(X)
        return self.regularizer * (
            1.0 / sigma.pow(2).clamp_min(1e-6)
        ).mean()

    def is_psd(self) -> bool:
        return False  # symmetric but not strictly PSD


class MahalanobisKernel(Kernel):
    """Anisotropic Gaussian with a learned PSD metric.

    ``k(x, x') = exp(-(x - x')^T A (x - x'))`` with
    ``A = L L^T`` for a learnable lower-triangular factor ``L``,
    ensuring ``A`` is PSD.

    With ``L = I`` the kernel reduces to the unit-bandwidth Gaussian,
    so the kernel is a strict generalisation of :class:`RBFKernel`
    that admits direction-dependent metrics without re-projecting
    the embeddings explicitly.
    """

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.L = nn.Parameter(torch.eye(d_in))

    def forward(
        self, X1: FeatureTensor, X2: FeatureTensor
    ) -> GramTensor:
        # Equivalent to RBF on L^T x:
        Z1 = X1 @ self.L
        Z2 = X2 @ self.L
        D2 = _pairwise_sq_dist(Z1, Z2)
        return torch.exp(-D2)

    def is_psd(self) -> bool:
        return True


class CauchyKernel(Kernel):
    """Cauchy / Student-t heavy-tailed kernel.

    ``k(x, x') = (1 + ||x - x'||^2 / nu)^{-(nu + 1) / 2}``.

    By Schoenberg's theorem the Gram matrix is PSD whenever
    ``nu >= d / 2`` where ``d`` is the ambient dimension; the default
    ``nu = 1`` is PSD in 1D and 2D, the regimes used by t-SNE.
    """

    def __init__(self, nu: float = 1.0) -> None:
        super().__init__()
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.nu = float(nu)

    def forward(
        self, X1: FeatureTensor, X2: FeatureTensor
    ) -> GramTensor:
        D2 = _pairwise_sq_dist(X1, X2)
        exponent = -(self.nu + 1.0) / 2.0
        return (1.0 + D2 / self.nu).pow(exponent)

    def is_psd(self) -> bool:
        # Schoenberg's condition is dimension-dependent; flagged True
        # to mark "PSD when nu is large enough"; consumers needing a
        # hard guarantee should check empirically.
        return True


class EpanechnikovKernel(Kernel):
    """Compact-support polynomial kernel.

    ``k(x, x') = max(0, 1 - r^2)^p`` with ``r = ||x - x'|| / h``.

    Variants:
        ``power=1``: Epanechnikov (the namesake).
        ``power=2``: biweight.
        ``power=3``: triweight.

    The kernel is *intrinsically sparse*: any pair beyond ``h`` in
    the embedding metric contributes exactly zero.  Downstream
    sparsification (:cref:`ch:05`) is therefore unnecessary.
    """

    def __init__(self, bandwidth: float = 1.0, power: int = 1) -> None:
        super().__init__()
        if bandwidth <= 0:
            raise ValueError(
                f"bandwidth must be positive, got {bandwidth}"
            )
        if power < 1:
            raise ValueError(f"power must be >= 1, got {power}")
        self.log_h = nn.Parameter(
            torch.tensor(float(bandwidth)).log()
        )
        self.power = int(power)

    def _h(self) -> torch.Tensor:
        return self.log_h.exp().clamp_min(1e-6)

    def forward(
        self, X1: FeatureTensor, X2: FeatureTensor
    ) -> GramTensor:
        D2 = _pairwise_sq_dist(X1, X2)
        h = self._h()
        r2 = D2 / (h * h)
        base = F.relu(1.0 - r2)
        if self.power == 1:
            return base
        return base.pow(self.power)

    def is_psd(self) -> bool:
        return True
