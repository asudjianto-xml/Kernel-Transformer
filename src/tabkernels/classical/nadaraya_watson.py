"""Nadaraya--Watson estimator (Chapter 2 §2.2).

Uses a user-supplied kernel function $k(x, x')$ to weight training points.
The kernel must be non-negative for the standard NW formula; if it isn't,
absolute values or a non-negative feature map should be used.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian RBF kernel: k(x, x') = exp(-||x-x'||^2 / sigma^2)."""
    sq = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
    return np.exp(-sq / (sigma ** 2)).astype(np.float32)


class NadarayaWatson:
    """Nadaraya-Watson kernel-smoothed regressor (Eq. 2.4).

    Parameters
    ----------
    kernel : callable
        Function (X1, X2) -> Gram matrix of shape (N1, N2) with non-negative entries.
    bandwidth : float, optional
        If `kernel` is the default RBF, sigma to use.  Ignored if `kernel` is custom.
    """

    def __init__(self, kernel: Optional[Callable] = None, bandwidth: float = 1.0):
        self.bandwidth = bandwidth
        self.kernel = kernel if kernel is not None else (
            lambda X1, X2: rbf_kernel(X1, X2, sigma=self.bandwidth)
        )
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NadarayaWatson":
        self._X = np.asarray(X, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32)
        return self

    def predict(self, X_q: np.ndarray) -> np.ndarray:
        if self._X is None:
            raise RuntimeError("fit() must be called before predict()")
        X_q = np.asarray(X_q, dtype=np.float32)
        if X_q.ndim == 1:
            X_q = X_q[None, :]
        K = self.kernel(X_q, self._X)  # (N_q, N)
        denom = K.sum(axis=1, keepdims=True)
        denom = np.where(denom > 1e-12, denom, 1.0)
        W = K / denom
        return (W @ self._y).astype(np.float32)

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Return the (N1, N2) Gram matrix between X1 and X2."""
        return self.kernel(np.asarray(X1, dtype=np.float32),
                           np.asarray(X2, dtype=np.float32))
