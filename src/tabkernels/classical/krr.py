"""Kernel ridge regression (Chapter 2 §2.3).

The minimiser of regularised squared loss in an RKHS, equivalent to the
posterior mean of GP regression with covariance kernel $k$ and noise
variance $N\\lambda$.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from tabkernels.classical.nadaraya_watson import rbf_kernel


class KernelRidgeRegression:
    """Kernel ridge regression (Eq. 2.7 in the dual form).

    Parameters
    ----------
    kernel : callable, optional
        Function (X1, X2) -> Gram matrix.  If None, uses RBF with `bandwidth`.
    bandwidth : float
        Bandwidth for the default RBF kernel.
    ridge : float
        Regularisation parameter $\\lambda$.  The dual coefficients are
        $\\alpha = (K + N\\lambda I)^{-1} y$.
    """

    def __init__(self, kernel: Optional[Callable] = None,
                 bandwidth: float = 1.0, ridge: float = 1.0):
        self.bandwidth = bandwidth
        self.ridge = ridge
        self.kernel = kernel if kernel is not None else (
            lambda X1, X2: rbf_kernel(X1, X2, sigma=self.bandwidth)
        )
        self._X: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        N = X.shape[0]
        K = self.kernel(X, X)  # (N, N)
        self._alpha = np.linalg.solve(K + N * self.ridge * np.eye(N, dtype=K.dtype), y)
        self._X = X
        return self

    def predict(self, X_q: np.ndarray) -> np.ndarray:
        if self._X is None:
            raise RuntimeError("fit() must be called before predict()")
        X_q = np.asarray(X_q, dtype=np.float32)
        if X_q.ndim == 1:
            X_q = X_q[None, :]
        K_q = self.kernel(X_q, self._X)  # (N_q, N)
        return (K_q @ self._alpha).astype(np.float32)

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Return the (N1, N2) Gram matrix between X1 and X2."""
        return self.kernel(np.asarray(X1, dtype=np.float32),
                           np.asarray(X2, dtype=np.float32))

    @property
    def dual_coefs(self) -> np.ndarray:
        """The dual coefficients $\\alpha = (K + N\\lambda I)^{-1} y$."""
        if self._alpha is None:
            raise RuntimeError("fit() must be called first")
        return self._alpha
