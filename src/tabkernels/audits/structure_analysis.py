"""Decomposition-recovery diagnostic on trained models (Chapter 13 §13.2-§13.4).

Migrated from ``~/jupyterlab/ICL/oracle_kernel_structure_analysis.py``.

For each ``(structure, task, seed)`` we train Std and Dual kernel-attention
variants (using the same anchor-based prior as :class:`OracleAudit`), extract
their learned bilinear form ``B``, decompose into ``B_S, B_A``, and compare
against the oracle's sample-space gram matrix.

Headline finding (cached in ``cached_audits/structure_analysis.json``): on the
sym+asym structure, Dual recovers the sym/skew split visibly while Std bakes
the structure into ``W_Q != W_K`` in a less interpretable way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from tabkernels.core.base import Audit

from ._helpers import (
    cosine_F_np,
    energy_split_np,
    eval_kernel_attention,
    train_kernel_attention,
)
from ._models import KernelAttention
from .base import AuditReport, save_report, seed_all
from .oracle_audit import make_data, oracle_features, oracle_predict

Regime = Literal["sym", "sym_asym"]
Task = Literal["reg", "cla"]


def std_kernel_matrix(model: KernelAttention) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``B = W_Q^T W_K`` from a Std :class:`KernelAttention`."""
    Wq = model.W_Q.weight.detach().cpu()
    Wk = model.W_K.weight.detach().cpu()
    B = Wq.T @ Wk
    B_S = (B + B.T) / 2
    B_A = (B - B.T) / 2
    return B.numpy(), B_S.numpy(), B_A.numpy()


def dual_kernel_matrix(model: KernelAttention) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``M_S, M_A`` from a Dual :class:`KernelAttention`."""
    M_S_raw = model.M_S.detach().cpu()
    M_A_raw = model.M_A.detach().cpu()
    B_S = ((M_S_raw + M_S_raw.T) / 2).numpy()
    B_A = ((M_A_raw - M_A_raw.T) / 2).numpy()
    return B_S + B_A, B_S, B_A


def kernel_gram_on_data(B: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pairwise gram ``K[i, j] = x_i^T B x_j`` on training points."""
    return X @ B @ X.T


# ----------------------------- Audit -----------------------------


class StructureAnalysis(Audit):
    """Train Std and Dual; report energy fractions and cosine to the oracle gram."""

    def __init__(self, *, N_total: int = 400, N_train: int = 200, D: int = 8,
                 n_seeds: int = 5,
                 structures: list[Regime] | None = None,
                 tasks: list[Task] | None = None,
                 d_emb: int = 16, epochs: int = 1500) -> None:
        self.N_total = N_total
        self.N_train = N_train
        self.D = D
        self.n_seeds = n_seeds
        self.structures: list[Regime] = list(structures or ["sym", "sym_asym"])
        self.tasks: list[Task] = list(tasks or ["reg", "cla"])
        self.d_emb = d_emb
        self.epochs = epochs

    def _analyze_one(self, structure: Regime, task: Task, seed: int) -> dict:
        seed_all(seed)
        data = make_data(N=self.N_total, D=self.D, regime=structure,
                         task=task, seed=seed)
        X = data["X"]
        y = data["y"]
        rng = np.random.RandomState(seed + 100)
        perm = rng.permutation(self.N_total)
        idx_tr = perm[:self.N_train]
        idx_te = perm[self.N_train:]
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_te, y_te = X[idx_te], y[idx_te]

        Phi_tr = oracle_features(
            X_tr, data["X_anchor"], data["v"],
            data["sigma"], data["sigma_a"], data["alpha"], structure,
        )
        G_oracle_train = Phi_tr @ Phi_tr.T
        sym_frac_oracle, asym_frac_oracle = energy_split_np(G_oracle_train)

        # Train Std
        torch.manual_seed(seed + 1000)
        std_model = KernelAttention(d_in=self.D, d_emb=self.d_emb,
                                    variant="std", aggregation="softmax")
        train_kernel_attention(std_model, X_tr, y_tr, task=task,
                               epochs=self.epochs)
        std_B, std_B_S, std_B_A = std_kernel_matrix(std_model)
        std_sym_frac, std_asym_frac = energy_split_np(std_B)
        G_std_train = kernel_gram_on_data(std_B, X_tr)
        G_std_S = (G_std_train + G_std_train.T) / 2
        G_std_sym_frac, G_std_asym_frac = energy_split_np(G_std_train)
        cos_std_oracle = cosine_F_np(G_std_S, G_oracle_train)

        # Train Dual
        torch.manual_seed(seed + 1000)
        dual_model = KernelAttention(d_in=self.D, d_emb=self.d_emb,
                                     variant="dual", aggregation="softmax")
        train_kernel_attention(dual_model, X_tr, y_tr, task=task,
                               epochs=self.epochs)
        dual_B, dual_B_S, dual_B_A = dual_kernel_matrix(dual_model)
        dual_sym_frac, dual_asym_frac = energy_split_np(dual_B)
        G_dual_train = kernel_gram_on_data(dual_B, X_tr)
        G_dual_S = (G_dual_train + G_dual_train.T) / 2
        cos_dual_oracle = cosine_F_np(G_dual_S, G_oracle_train)

        std_eval = eval_kernel_attention(std_model, X_tr, y_tr, X_te, y_te, task=task)
        dual_eval = eval_kernel_attention(dual_model, X_tr, y_tr, X_te, y_te, task=task)
        ora = oracle_predict(data, idx_tr, idx_te, structure, task)

        return {
            "structure": structure,
            "task": task,
            "seed": seed,
            "oracle": ora,
            "std": {
                "test": std_eval,
                "B_sym_frac": std_sym_frac,
                "B_asym_frac": std_asym_frac,
                "G_sym_frac": G_std_sym_frac,
                "G_asym_frac": G_std_asym_frac,
                "cos_to_oracle": cos_std_oracle,
            },
            "dual": {
                "test": dual_eval,
                "B_sym_frac": dual_sym_frac,
                "B_asym_frac": dual_asym_frac,
                "cos_to_oracle": cos_dual_oracle,
                "M_S_norm": float(np.linalg.norm(dual_B_S)),
                "M_A_norm": float(np.linalg.norm(dual_B_A)),
            },
            "oracle_sample_gram": {
                "sym_frac": sym_frac_oracle,
                "asym_frac": asym_frac_oracle,
            },
        }

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        per_seed: list[dict] = []
        for structure in self.structures:
            for task in self.tasks:
                for seed in range(self.n_seeds):
                    per_seed.append(self._analyze_one(structure, task, seed))

        report: AuditReport = {
            "config": {
                "N_total": self.N_total, "N_train": self.N_train, "D": self.D,
                "n_seeds": self.n_seeds, "structures": self.structures,
                "tasks": self.tasks, "d_emb": self.d_emb,
                "epochs": self.epochs,
            },
            "metrics": self._aggregate(per_seed),
            "per_seed": per_seed,
            "decomposition": self._decomposition_summary(per_seed),
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    def _aggregate(self, rows: list[dict]) -> dict:
        out: dict = {}
        for structure in self.structures:
            for task in self.tasks:
                metric = "mse" if task == "reg" else "acc"
                ents = [r for r in rows if r["structure"] == structure
                        and r["task"] == task]
                if not ents:
                    continue
                std_test = np.array([e["std"]["test"][metric] for e in ents])
                dual_test = np.array([e["dual"]["test"][metric] for e in ents])
                ora = np.array([e["oracle"][metric] for e in ents])
                out[f"{structure}_{task}"] = {
                    "metric": metric,
                    "oracle_mean": float(ora.mean()),
                    "std_mean": float(std_test.mean()),
                    "dual_mean": float(dual_test.mean()),
                }
        return out

    def _decomposition_summary(self, rows: list[dict]) -> dict:
        out: dict = {}
        for structure in self.structures:
            for task in self.tasks:
                ents = [r for r in rows if r["structure"] == structure
                        and r["task"] == task]
                if not ents:
                    continue
                std_B_sym = np.array([e["std"]["B_sym_frac"] for e in ents])
                dual_B_sym = np.array([e["dual"]["B_sym_frac"] for e in ents])
                dual_M_S_norm = np.array([e["dual"]["M_S_norm"] for e in ents])
                dual_M_A_norm = np.array([e["dual"]["M_A_norm"] for e in ents])
                out[f"{structure}_{task}"] = {
                    "std_B_sym_frac_mean": float(std_B_sym.mean()),
                    "dual_B_sym_frac_mean": float(dual_B_sym.mean()),
                    "dual_M_S_norm_mean": float(dual_M_S_norm.mean()),
                    "dual_M_A_norm_mean": float(dual_M_A_norm.mean()),
                    "dual_M_A_over_M_S": float(
                        (dual_M_A_norm / (dual_M_S_norm + 1e-12)).mean()
                    ),
                }
        return out


__all__ = [
    "StructureAnalysis",
    "std_kernel_matrix",
    "dual_kernel_matrix",
    "kernel_gram_on_data",
]
