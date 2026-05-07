"""$k$-nearest-neighbour predictors (Chapter 2 §2.1).

Pure-numpy reference implementations used as the starting point for the kernel
family throughout the book.  These predictors implement scikit-learn's
fit/predict interface so they compose with sklearn pipelines.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class KNNRegressor:
    """$k$-nearest-neighbour regressor with uniform or distance weights.

    Parameters
    ----------
    k : int
        Number of neighbours to average over.
    weights : {'uniform', 'distance'}
        Uniform: simple average over neighbours (Eq. 2.2).
        Distance: weights inversely proportional to distance (a one-step
        bridge to Nadaraya-Watson; see Section 2.2).
    """

    def __init__(self, k: int = 5, weights: str = "uniform"):
        if k < 1:
            raise ValueError("k must be >= 1")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")
        self.k = k
        self.weights = weights
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be 1D with len {X.shape[0]}, got shape {y.shape}")
        if self.k > X.shape[0]:
            raise ValueError(f"k={self.k} exceeds N={X.shape[0]}")
        self._X = X
        self._y = y
        return self

    def predict(self, X_q: np.ndarray) -> np.ndarray:
        if self._X is None:
            raise RuntimeError("fit() must be called before predict()")
        X_q = np.asarray(X_q, dtype=np.float32)
        if X_q.ndim == 1:
            X_q = X_q[None, :]
        # Pairwise squared distances.
        sq = ((X_q[:, None, :] - self._X[None, :, :]) ** 2).sum(-1)
        # Indices of k smallest distances per query.
        idx = np.argpartition(sq, kth=self.k - 1, axis=1)[:, : self.k]
        if self.weights == "uniform":
            preds = self._y[idx].mean(axis=1)
        else:
            d = np.take_along_axis(sq, idx, axis=1) ** 0.5
            w = 1.0 / np.clip(d, 1e-12, None)
            w = w / w.sum(axis=1, keepdims=True)
            preds = (w * self._y[idx]).sum(axis=1)
        return preds.astype(np.float32)


class KNNClassifier:
    """$k$-nearest-neighbour classifier with majority vote or distance-weighted vote."""

    def __init__(self, k: int = 5, weights: str = "uniform"):
        if k < 1:
            raise ValueError("k must be >= 1")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")
        self.k = k
        self.weights = weights
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self._classes = np.unique(y)
        self._X = X
        self._y = y
        return self

    def predict(self, X_q: np.ndarray) -> np.ndarray:
        if self._X is None:
            raise RuntimeError("fit() must be called before predict()")
        X_q = np.asarray(X_q, dtype=np.float32)
        if X_q.ndim == 1:
            X_q = X_q[None, :]
        sq = ((X_q[:, None, :] - self._X[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(sq, kth=self.k - 1, axis=1)[:, : self.k]
        if self.weights == "uniform":
            votes = np.zeros((X_q.shape[0], len(self._classes)))
            for i, cls in enumerate(self._classes):
                votes[:, i] = (self._y[idx] == cls).sum(axis=1)
        else:
            d = np.take_along_axis(sq, idx, axis=1) ** 0.5
            w = 1.0 / np.clip(d, 1e-12, None)
            votes = np.zeros((X_q.shape[0], len(self._classes)))
            for i, cls in enumerate(self._classes):
                votes[:, i] = (w * (self._y[idx] == cls)).sum(axis=1)
        winners = votes.argmax(axis=1)
        return self._classes[winners]
