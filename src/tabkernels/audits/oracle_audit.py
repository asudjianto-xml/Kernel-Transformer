"""Supervised oracle audit (Chapter 10 §10.6).

Migrated from ``~/jupyterlab/ICL/oracle_kernel_audit_v2.py``.

Generates synthetic data with controlled kernel structure (anchor-based
RBF + optional skew-symmetric ``tanh`` correction). Trains five
:class:`tabkernels.audits._models.KernelAttention` variants and compares them
against the closed-form oracle predictor that uses the true generating
feature map.

Headline finding (cached in ``cached_audits/oracle_audit.json``): on the
sym-only structure all symmetric variants approach the oracle; on the
sym+asym structure the Std and Dual variants dominate while the pure-asym
variant is destructive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from tabkernels.core.base import Audit

from ._helpers import eval_kernel_attention, train_kernel_attention
from ._models import KernelAttention
from .base import AuditReport, save_report, seed_all

Regime = Literal["sym", "sym_asym"]
Task = Literal["reg", "cla"]


# ----------------------------- Data -----------------------------


def make_data(N: int = 400, D: int = 8, *, regime: Regime = "sym",
              task: Task = "reg", M: int = 20, alpha: float = 1.0,
              sigma: float = 2.0, sigma_a: float = 1.0, noise: float = 0.1,
              seed: int = 0) -> dict:
    """Generate ``X`` and ``y = f(X) + noise`` with controlled structure.

    Returns a dict containing ``X``, ``y``, the generating feature map
    ``Phi``, the random anchors and (for the sym+asym regime) the asymmetry
    direction ``v``.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D).astype(np.float32)
    X_anchor = rng.randn(M, D).astype(np.float32)
    c = rng.randn(M).astype(np.float32)
    if regime == "sym_asym":
        v_asym = rng.randn(D).astype(np.float32)
        v_asym /= np.linalg.norm(v_asym)
    else:
        v_asym = None

    def features(X_set: np.ndarray) -> np.ndarray:
        diff = X_set[:, None, :] - X_anchor[None, :, :]
        sq = (diff ** 2).sum(-1)
        f_S = np.exp(-sq / sigma ** 2).astype(np.float32)
        if regime == "sym":
            return f_S
        if regime == "sym_asym":
            proj = (diff @ v_asym).astype(np.float32)
            f_A = np.tanh(proj / sigma_a).astype(np.float32)
            return f_S + alpha * f_A
        raise ValueError(regime)

    Phi = features(X)
    f = Phi @ c
    f = (f - f.mean()) / (f.std() + 1e-8)

    if task == "reg":
        y = (f + noise * rng.randn(N)).astype(np.float32)
    elif task == "cla":
        y = (f > 0).astype(np.float32)
    else:
        raise ValueError(task)

    return dict(
        X=X, y=y, Phi=Phi, c=c, X_anchor=X_anchor, v=v_asym,
        sigma=sigma, sigma_a=sigma_a, alpha=alpha, regime=regime, task=task,
    )


def oracle_features(X: np.ndarray, X_anchor: np.ndarray, v: np.ndarray | None,
                    sigma: float, sigma_a: float, alpha: float,
                    regime: Regime) -> np.ndarray:
    """Recompute the generating feature map for arbitrary ``X``."""
    diff = X[:, None, :] - X_anchor[None, :, :]
    sq = (diff ** 2).sum(-1)
    f_S = np.exp(-sq / sigma ** 2).astype(np.float32)
    if regime == "sym":
        return f_S
    proj = (diff @ v).astype(np.float32)
    f_A = np.tanh(proj / sigma_a).astype(np.float32)
    return f_S + alpha * f_A


def oracle_predict(data: dict, idx_train: np.ndarray, idx_test: np.ndarray,
                   regime: Regime, task: Task,
                   ridge: float = 0.05) -> dict:
    """Oracle predictor: ridge / logistic regression on the true feature map."""
    Phi = oracle_features(
        data["X"], data["X_anchor"], data["v"],
        data["sigma"], data["sigma_a"], data["alpha"], regime,
    )
    Phi_tr = Phi[idx_train]
    Phi_te = Phi[idx_test]
    y_tr = data["y"][idx_train]
    y_te = data["y"][idx_test]
    M = Phi.shape[1]
    if task == "reg":
        A = Phi_tr.T @ Phi_tr + ridge * np.eye(M, dtype=np.float32)
        b = Phi_tr.T @ y_tr
        c_hat = np.linalg.solve(A, b)
        y_pred = Phi_te @ c_hat
        return {"mse": float(np.mean((y_pred - y_te) ** 2))}
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    lr = LogisticRegression(C=1.0 / ridge, max_iter=200,
                            solver="lbfgs").fit(Phi_tr, y_tr)
    y_pred = lr.predict(Phi_te)
    return {"acc": float(np.mean(y_pred == y_te))}


# ----------------------------- Audit -----------------------------


class OracleAudit(Audit):
    """Train the five kernel-attention variants under two structures × two tasks.

    Returns a report with one ``per_seed`` row per ``(regime, task, seed)``,
    plus aggregate ``metrics`` of mean/std test MSE/acc per variant and
    oracle gap.
    """

    def __init__(self, *, N_total: int = 400, N_train: int = 200, D: int = 8,
                 n_seeds: int = 5,
                 regimes: list[Regime] | None = None,
                 tasks: list[Task] | None = None,
                 variants: list[str] | None = None,
                 nw_variants: list[str] | None = None,
                 d_emb: int = 16, epochs: int = 1500) -> None:
        self.N_total = N_total
        self.N_train = N_train
        self.D = D
        self.n_seeds = n_seeds
        self.regimes: list[Regime] = list(regimes or ["sym", "sym_asym"])
        self.tasks: list[Task] = list(tasks or ["reg", "cla"])
        self.variants = list(variants or ["std", "sym_psd", "sym_gen",
                                          "pure_asym", "dual"])
        self.nw_variants = list(nw_variants or ["std", "sym_psd"])
        self.d_emb = d_emb
        self.epochs = epochs

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        """Run the audit and return a structured :class:`AuditReport`."""
        per_seed: list[dict] = []
        for regime in self.regimes:
            for task in self.tasks:
                for seed in range(self.n_seeds):
                    seed_all(seed)
                    data = make_data(N=self.N_total, D=self.D, regime=regime,
                                     task=task, seed=seed)
                    X = data["X"]
                    y = data["y"]
                    rng = np.random.RandomState(seed + 100)
                    perm = rng.permutation(self.N_total)
                    idx_tr = perm[:self.N_train]
                    idx_te = perm[self.N_train:]
                    X_tr, y_tr = X[idx_tr], y[idx_tr]
                    X_te, y_te = X[idx_te], y[idx_te]

                    ora = oracle_predict(data, idx_tr, idx_te, regime, task)
                    variants_results: dict = {}
                    row: dict = {
                        "structure": regime, "task": task, "seed": seed,
                        "oracle": ora, "variants": variants_results,
                    }

                    for v in self.variants:
                        torch.manual_seed(seed + 1000)
                        m = KernelAttention(d_in=self.D, d_emb=self.d_emb,
                                            variant=v, aggregation="softmax")
                        train_kernel_attention(m, X_tr, y_tr, task=task,
                                               epochs=self.epochs)
                        res = eval_kernel_attention(m, X_tr, y_tr, X_te, y_te,
                                                    task=task)
                        variants_results[f"{v}_softmax"] = res

                    for v in self.nw_variants:
                        torch.manual_seed(seed + 1000)
                        m = KernelAttention(d_in=self.D, d_emb=self.d_emb,
                                            variant=v,
                                            aggregation="nw_performer")
                        train_kernel_attention(m, X_tr, y_tr, task=task,
                                               epochs=self.epochs)
                        res = eval_kernel_attention(m, X_tr, y_tr, X_te, y_te,
                                                    task=task)
                        variants_results[f"{v}_nw"] = res

                    per_seed.append(row)

        metrics = self._aggregate(per_seed)
        report: AuditReport = {
            "config": {
                "N_total": self.N_total, "N_train": self.N_train, "D": self.D,
                "n_seeds": self.n_seeds, "regimes": self.regimes,
                "tasks": self.tasks, "variants": self.variants,
                "nw_variants": self.nw_variants, "d_emb": self.d_emb,
                "epochs": self.epochs,
            },
            "metrics": metrics,
            "per_seed": per_seed,
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        all_keys = sorted({k for r in rows for k in r["variants"]})
        out: dict = {}
        regimes = sorted({r["structure"] for r in rows})
        tasks = sorted({r["task"] for r in rows})
        for regime in regimes:
            for task in tasks:
                metric = "mse" if task == "reg" else "acc"
                ents = [r for r in rows if r["structure"] == regime
                        and r["task"] == task]
                if not ents:
                    continue
                ora = np.array([e["oracle"][metric] for e in ents])
                key = f"{regime}_{task}"
                v_summary: dict = {}
                summary: dict = {
                    "metric": metric,
                    "oracle_mean": float(ora.mean()),
                    "oracle_std": float(ora.std()),
                    "variants": v_summary,
                }
                for vk in all_keys:
                    vals = np.array([e["variants"][vk][metric] for e in ents])
                    v_summary[vk] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "delta_vs_oracle": float((vals - ora).mean()),
                    }
                out[key] = summary
        return out


__all__ = ["OracleAudit", "make_data", "oracle_features", "oracle_predict"]
