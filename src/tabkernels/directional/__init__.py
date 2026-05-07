"""Directional generators and causal-kernel demos (Chapter 25).

Wraps the lower-level generators in :mod:`tabkernels.audits.directional`
behind dataclass-style classes, and provides a small toy demo of
kernel-based treatment-effect estimation.

Generators
----------
:class:`DAGGenerator`        — random DAG with linear-plus-mean-of-parents structure.
:class:`ARGenerator`         — autoregressive time series with sinusoidal features.

Demo
----
:func:`causal_kernel_inference` — kernel-regression-based ATE estimator on a
synthetic 2-arm treatment dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tabkernels.audits.directional import (
    make_ar_data,
    make_dag_data,
    oracle_predict_ar,
    oracle_predict_dag,
)


@dataclass
class DAGGenerator:
    """Random DAG generator: ``y_v = alpha * mean(y_pa(v)) + beta . x_v + noise``."""
    D: int = 8
    max_parents: int = 3
    alpha: float = 0.7
    noise: float = 0.1

    def sample(self, N: int = 200, seed: int = 0) -> dict:
        return make_dag_data(N=N, D=self.D, max_parents=self.max_parents,
                             alpha=self.alpha, noise=self.noise, seed=seed)

    def oracle(self, data: dict, idx_train, idx_test):
        return oracle_predict_dag(data, idx_train, idx_test)


@dataclass
class ARGenerator:
    """AR(p) time-series generator with sinusoidal feature columns."""
    D: int = 8
    p: int = 2
    noise: float = 0.1

    def sample(self, T: int = 300, seed: int = 0) -> dict:
        return make_ar_data(T=T, D=self.D, p=self.p, noise=self.noise, seed=seed)

    def oracle(self, data: dict, idx_train, idx_test):
        return oracle_predict_ar(data, idx_train, idx_test)


# ----------------------------- Causal-effect demo -----------------------------


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    sq = ((X1[:, None] - X2[None, :]) ** 2).sum(-1)
    return np.exp(-sq / (sigma ** 2))


def make_treatment_effect_data(N: int = 400, d: int = 4, tau_strength: float = 1.0,
                               noise: float = 0.2, seed: int = 0) -> dict:
    """Synthetic 2-arm treatment dataset with known heterogeneous effect.

    Generative process:
      X ~ N(0, I_d)
      pi(X) = sigmoid(X[:, 0])  (selection on first feature)
      A ~ Bernoulli(pi(X))
      Y(0) = f0(X) + eps0
      Y(1) = Y(0) + tau(X)
      Y    = (1-A) Y(0) + A Y(1)
    where f0 is a smooth function and tau(X) = tau_strength * (X[:, 0] + 0.5).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(N, d).astype(np.float32)
    f0 = (np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] - 0.3 * X[:, 2]).astype(np.float32)
    tau = tau_strength * (X[:, 0] + 0.5).astype(np.float32)
    pi = 1.0 / (1.0 + np.exp(-X[:, 0]))
    A = (rng.rand(N) < pi).astype(np.int64)
    eps0 = noise * rng.randn(N).astype(np.float32)
    eps1 = noise * rng.randn(N).astype(np.float32)
    Y0 = f0 + eps0
    Y1 = f0 + tau + eps1
    Y = np.where(A == 1, Y1, Y0).astype(np.float32)
    return {"X": X, "A": A, "Y": Y, "tau": tau, "true_ate": float(tau.mean()),
            "f0": f0, "Y0": Y0, "Y1": Y1}


def _kernel_regression(X_train: np.ndarray, y_train: np.ndarray,
                       X_query: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    K = _rbf_kernel(X_query, X_train, sigma=sigma)
    K = K / K.sum(axis=1, keepdims=True).clip(1e-9)
    return K @ y_train


def causal_kernel_inference(data: dict, sigma: float = 1.0) -> dict:
    """Estimate the average treatment effect using kernel regression on each arm.

    Naive estimator: separate kernel regressions on each treatment arm,
    evaluated at every unit's covariates, then averaged.
    """
    X = data["X"]; A = data["A"]; Y = data["Y"]
    idx0 = np.where(A == 0)[0]
    idx1 = np.where(A == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("both treatment arms must be non-empty")
    yhat0_at_all = _kernel_regression(X[idx0], Y[idx0], X, sigma=sigma)
    yhat1_at_all = _kernel_regression(X[idx1], Y[idx1], X, sigma=sigma)
    cate = yhat1_at_all - yhat0_at_all
    return {
        "ate_hat": float(cate.mean()),
        "cate_hat": cate,
        "true_ate": data["true_ate"],
        "true_cate": data["tau"],
    }


__all__ = [
    "DAGGenerator",
    "ARGenerator",
    "make_treatment_effect_data",
    "causal_kernel_inference",
]
