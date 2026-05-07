"""Directional-data audit (Chapter 10 §10.7).

Migrated from ``~/jupyterlab/ICL/directional_data_audit.py`` and
``~/jupyterlab/ICL/directional_timeordered.py``.

* :class:`DirectionalAudit` — DAG and AR(p) generators with random splits.
* :class:`TimeOrderedAuditAR` — AR(p) with time-ordered split, evaluated
  both in batch and autoregressive modes.

Headline finding (cached in ``cached_audits/directional_random.json`` and
``directional_timeordered.json``): a causal mask is the only thing that
helps; kernel asymmetry on its own does not buy directionality.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from tabkernels.core.base import Audit

from ._helpers import (
    eval_kernel_attention,
    get_device,
    train_kernel_attention,
)
from ._models import KernelAttention
from .base import AuditReport, save_report, seed_all

Generator = Literal["dag", "ar"]


# ----------------------------- Data generators -----------------------------


def make_dag_data(N: int = 200, D: int = 8, max_parents: int = 3,
                  alpha: float = 0.7, noise: float = 0.1, seed: int = 0) -> dict:
    """Random DAG: ``y_v = alpha * mean(y_pa(v)) + beta . x_v + noise``."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D).astype(np.float32)
    X[:, 0] = (np.arange(N) / N).astype(np.float32)

    parents: list[np.ndarray] = [np.array([], dtype=int) for _ in range(N)]
    y = np.zeros(N, dtype=np.float32)
    beta = (rng.randn(D) * 0.5).astype(np.float32)

    for v in range(N):
        if v == 0:
            y[v] = float(beta @ X[v] + noise * rng.randn())
        else:
            n_par = int(rng.randint(1, min(max_parents, v) + 1))
            parents[v] = rng.choice(v, size=n_par, replace=False)
            y[v] = float(alpha * y[parents[v]].mean()
                         + beta @ X[v] + noise * rng.randn())
    order = np.arange(N).astype(np.int64)
    return dict(X=X, y=y, parents=parents, alpha=alpha, beta=beta,
                order=order, kind="dag")


def make_ar_data(T: int = 300, D: int = 8, p: int = 2,
                 noise: float = 0.1, seed: int = 0) -> dict:
    """Random AR(``p``) time series with sinusoidal feature columns."""
    rng = np.random.RandomState(seed)
    X = rng.randn(T, D).astype(np.float32)
    t_norm = (np.arange(T) / T).astype(np.float32)
    X[:, 0] = np.sin(2 * np.pi * t_norm)
    X[:, 1] = np.cos(2 * np.pi * t_norm)

    rho = rng.uniform(-0.5, 0.5, size=p).astype(np.float32)
    beta = (rng.randn(D) * 0.3).astype(np.float32)

    y = np.zeros(T, dtype=np.float32)
    for t in range(T):
        ar_part = 0.0
        for s in range(p):
            if t - s - 1 >= 0:
                ar_part += rho[s] * y[t - s - 1]
        y[t] = float(ar_part + beta @ X[t] + noise * rng.randn())
    order = np.arange(T).astype(np.int64)
    return dict(X=X, y=y, rho=rho, beta=beta, order=order, kind="ar", p=p)


# ----------------------------- Oracles -----------------------------


def oracle_predict_dag(data: dict, idx_train: np.ndarray,
                       idx_test: np.ndarray) -> np.ndarray:
    alpha = data["alpha"]
    beta = data["beta"]
    train_set = set(idx_train.tolist())
    pred: dict[int, float] = {}
    for v in sorted(idx_test.tolist()):
        if len(data["parents"][v]) > 0:
            ys = []
            for u in data["parents"][v]:
                if u in train_set:
                    ys.append(data["y"][u])
                else:
                    ys.append(pred.get(u, 0.0))
            y_pa = float(np.mean(ys))
        else:
            y_pa = 0.0
        pred[v] = float(alpha * y_pa + beta @ data["X"][v])
    return np.array([pred[v] for v in idx_test], dtype=np.float32)


def oracle_predict_ar(data: dict, idx_train: np.ndarray,
                      idx_test: np.ndarray) -> np.ndarray:
    rho = data["rho"]
    beta = data["beta"]
    p = data["p"]
    train_set = set(idx_train.tolist())
    pred: dict[int, float] = {}
    for t in sorted(idx_test.tolist()):
        ar = 0.0
        for s in range(p):
            tau = t - s - 1
            if tau < 0:
                continue
            if tau in train_set:
                ar += rho[s] * data["y"][tau]
            elif tau in pred:
                ar += rho[s] * pred[tau]
        pred[t] = float(ar + beta @ data["X"][t])
    return np.array([pred[t] for t in idx_test], dtype=np.float32)


# ----------------------------- DirectionalAudit -----------------------------


class DirectionalAudit(Audit):
    """Random-split DAG and AR(p) audit with mask / no-mask × Std / SymPSD."""

    _VARIANTS = (
        ("std",      "std",     False),
        ("sym_psd",  "sym_psd", False),
        ("std_mask", "std",     True),
        ("sym_mask", "sym_psd", True),
    )

    def __init__(self, *, n_seeds: int = 5, N_dag: int = 200, T_ar: int = 300,
                 D: int = 8, d_emb: int = 16, epochs: int = 2000) -> None:
        self.n_seeds = n_seeds
        self.N_dag = N_dag
        self.T_ar = T_ar
        self.D = D
        self.d_emb = d_emb
        self.epochs = epochs

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        sweeps: list[tuple[str, Any, Any, dict]] = [
            ("dag", make_dag_data, oracle_predict_dag,
             dict(N=self.N_dag, D=self.D)),
            ("ar",  make_ar_data,  oracle_predict_ar,
             dict(T=self.T_ar, D=self.D)),
        ]
        per_seed: list[dict] = []
        for kind, gen, oracle_fn, kw in sweeps:
            N_tot = kw.get("N") or kw.get("T")
            assert N_tot is not None
            N_tr = N_tot // 2
            for seed in range(self.n_seeds):
                seed_all(seed)
                data = gen(seed=seed, **kw)
                X = data["X"]
                y = data["y"]
                order = data["order"]
                rng = np.random.RandomState(seed + 100)
                perm = rng.permutation(N_tot)
                idx_tr = np.sort(perm[:N_tr])
                idx_te = np.sort(perm[N_tr:])
                X_tr, y_tr, ord_tr = X[idx_tr], y[idx_tr], order[idx_tr]
                X_te, y_te, ord_te = X[idx_te], y[idx_te], order[idx_te]
                ora_pred = oracle_fn(data, idx_tr, idx_te)
                ora_mse = float(np.mean((ora_pred - y_te) ** 2))
                v_results: dict = {}
                row: dict = {
                    "kind": kind, "seed": seed,
                    "y_var": float(np.var(y_te)),
                    "oracle_mse": ora_mse,
                    "variants": v_results,
                }
                for tag, variant, causal in self._VARIANTS:
                    torch.manual_seed(seed + 1000)
                    m = KernelAttention(d_in=X.shape[1], d_emb=self.d_emb,
                                        variant=variant, aggregation="softmax")
                    train_kernel_attention(
                        m, X_tr, y_tr, task="reg",
                        epochs=self.epochs, causal=causal,
                        order_train=ord_tr,
                    )
                    res = eval_kernel_attention(
                        m, X_tr, y_tr, X_te, y_te, task="reg",
                        causal=causal,
                        order_train=ord_tr, order_test=ord_te,
                    )
                    v_results[tag] = float(res["mse"])
                per_seed.append(row)

        metrics = self._aggregate(per_seed)
        report: AuditReport = {
            "config": {
                "n_seeds": self.n_seeds, "N_dag": self.N_dag,
                "T_ar": self.T_ar, "D": self.D, "d_emb": self.d_emb,
                "epochs": self.epochs,
                "variants": [t for t, _, _ in self._VARIANTS],
            },
            "metrics": metrics,
            "per_seed": per_seed,
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        out: dict = {}
        for kind in sorted({r["kind"] for r in rows}):
            ents = [r for r in rows if r["kind"] == kind]
            ora = np.array([e["oracle_mse"] for e in ents])
            yvar = np.array([e["y_var"] for e in ents])
            summary: dict = {
                "y_var_mean": float(yvar.mean()),
                "oracle_mse_mean": float(ora.mean()),
                "oracle_R2": float(1 - ora.mean() / (yvar.mean() + 1e-12)),
                "variants": {},
            }
            keys = sorted({k for e in ents for k in e["variants"]})
            for k in keys:
                vals = np.array([e["variants"][k] for e in ents])
                summary["variants"][k] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "delta_vs_oracle": float((vals - ora).mean()),
                }
            out[kind] = summary
        return out


# ----------------------------- TimeOrderedAuditAR -----------------------------


def _autoregressive_predict(model, X_tr, y_tr, ord_tr, X_te, ord_te, causal):
    device = get_device()
    model.eval()
    X_pool = list(X_tr)
    y_pool = list(y_tr)
    ord_pool = list(ord_tr)
    preds = np.zeros(len(X_te), dtype=np.float32)
    sort_idx = np.argsort(ord_te)
    for i in sort_idx:
        x_q = X_te[i:i + 1]
        o_q = ord_te[i:i + 1]
        Xp = torch.tensor(np.array(X_pool, dtype=np.float32), device=device)
        yp = torch.tensor(np.array(y_pool, dtype=np.float32), device=device)
        op = torch.tensor(np.array(ord_pool, dtype=np.float32), device=device)
        xq_t = torch.tensor(x_q.astype(np.float32), device=device)
        oq_t = torch.tensor(o_q.astype(np.float32), device=device)
        with torch.no_grad():
            y_pred = model.predict(
                xq_t, Xp, yp,
                order_q=oq_t, order_t=op,
                causal=causal, mask_diag=False,
            ).item()
        preds[i] = y_pred
        X_pool.append(x_q[0])
        y_pool.append(y_pred)
        ord_pool.append(o_q[0])
    return preds


class TimeOrderedAuditAR(Audit):
    """AR(p) with time-ordered train/test split (proper forecasting setup)."""

    _VARIANTS = (
        ("std",       "std",     False),
        ("sym_psd",   "sym_psd", False),
        ("std_mask",  "std",     True),
        ("sym_mask",  "sym_psd", True),
    )

    def __init__(self, *, n_seeds: int = 5, T: int = 400, D: int = 8,
                 p: int = 2, d_emb: int = 16, epochs: int = 2000) -> None:
        self.n_seeds = n_seeds
        self.T = T
        self.D = D
        self.p = p
        self.d_emb = d_emb
        self.epochs = epochs

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        train_n = self.T // 2
        per_seed: list[dict] = []
        for seed in range(self.n_seeds):
            seed_all(seed)
            data = make_ar_data(T=self.T, D=self.D, p=self.p, seed=seed)
            X = data["X"]
            y = data["y"]
            order = data["order"]
            idx_tr = np.arange(train_n)
            idx_te = np.arange(train_n, self.T)
            X_tr, y_tr, ord_tr = X[idx_tr], y[idx_tr], order[idx_tr]
            X_te, y_te, ord_te = X[idx_te], y[idx_te], order[idx_te]
            ora_pred = oracle_predict_ar(data, idx_tr, idx_te)
            ora_mse = float(np.mean((ora_pred - y_te) ** 2))
            row: dict = {
                "seed": seed,
                "y_var": float(np.var(y_te)),
                "oracle_mse": ora_mse,
                "variants": {},
            }
            for tag, variant, causal in self._VARIANTS:
                torch.manual_seed(seed + 1000)
                m = KernelAttention(d_in=X.shape[1], d_emb=self.d_emb,
                                    variant=variant, aggregation="softmax")
                train_kernel_attention(
                    m, X_tr, y_tr, task="reg",
                    epochs=self.epochs, causal=causal, order_train=ord_tr,
                )
                # batch
                res = eval_kernel_attention(
                    m, X_tr, y_tr, X_te, y_te, task="reg", causal=causal,
                    order_train=ord_tr, order_test=ord_te,
                )
                batch_mse = float(res["mse"])
                # autoregressive
                ar_pred = _autoregressive_predict(
                    m, X_tr, y_tr, ord_tr, X_te, ord_te, causal=causal,
                )
                ar_mse = float(np.mean((ar_pred - y_te) ** 2))
                row["variants"][tag] = {"batch_mse": batch_mse,
                                         "ar_mse": ar_mse}
            per_seed.append(row)

        metrics = self._aggregate(per_seed)
        report: AuditReport = {
            "config": {
                "n_seeds": self.n_seeds, "T": self.T, "D": self.D,
                "p": self.p, "d_emb": self.d_emb, "epochs": self.epochs,
                "variants": [t for t, _, _ in self._VARIANTS],
            },
            "metrics": metrics,
            "per_seed": per_seed,
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        out: dict = {}
        for mode in ("batch_mse", "ar_mse"):
            keys = sorted({k for r in rows for k in r["variants"]})
            mode_summary: dict = {}
            for k in keys:
                vals = np.array([r["variants"][k][mode] for r in rows])
                mode_summary[k] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                }
            out[mode] = mode_summary
        ora = np.array([r["oracle_mse"] for r in rows])
        out["oracle_mse_mean"] = float(ora.mean())
        return out


__all__ = [
    "DirectionalAudit",
    "TimeOrderedAuditAR",
    "make_dag_data",
    "make_ar_data",
    "oracle_predict_dag",
    "oracle_predict_ar",
]
