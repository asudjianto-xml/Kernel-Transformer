"""Pairwise prediction audit with controlled asymmetric ground truth (Chapter 11).

Migrated from ``~/jupyterlab/ICL/asymmetric_needed_audit.py`` (test MSE only)
and ``~/jupyterlab/ICL/asymmetric_decomposition.py`` (test MSE +
decomposition diagnostics).

:class:`PairwiseAudit` runs a sweep over five kernel parameterisations
(``sym_psd``, ``sym_gen``, ``pure_asym``, ``full``, ``dual``) on two ground-
truth regimes (``symmetric``, ``asymmetric``) and reports both held-out MSE
and decomposition recovery (Frobenius energy split + cosine similarity to
truth).

Headline finding: when the ground truth is asymmetric the symmetric
parameterisations leave the skew component on the table; ``full`` and
``dual`` recover both halves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from tabkernels.core.base import Audit

from ._helpers import cosine_F_np, energy_split_np, eval_pairwise, train_pairwise
from ._models import PairwisePredictor
from .base import AuditReport, save_report, seed_all

Regime = Literal["symmetric", "asymmetric"]


# ----------------------------- Data -----------------------------


def make_pairwise_data(N: int = 100, D: int = 8, *,
                       regime: Regime = "asymmetric",
                       noise: float = 0.05, seed: int = 0) -> dict:
    """``R_{ij} = z_i^T A z_j + noise`` with ``A`` symmetric or unconstrained."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, D).astype(np.float32)
    M = rng.randn(D, D).astype(np.float32)
    if regime == "symmetric":
        A = (M + M.T) / 2
    elif regime == "asymmetric":
        A = M
    else:
        raise ValueError(regime)
    A = A.astype(np.float32)
    R = Z @ A @ Z.T
    R = R / (R.std() + 1e-8)
    R_noisy = (R + noise * rng.randn(N, N)).astype(np.float32)
    return dict(Z=Z, R=R_noisy, R_clean=R, A_true=A, regime=regime)


# ----------------------------- Audit -----------------------------


class PairwiseAudit(Audit):
    """Pairwise prediction audit with optional decomposition diagnostics."""

    def __init__(self, *, n_seeds: int = 5, N: int = 100, D: int = 8,
                 regimes: list[Regime] | None = None,
                 variants: list[str] | None = None,
                 epochs: int = 2000,
                 with_decomposition: bool = True) -> None:
        self.n_seeds = n_seeds
        self.N = N
        self.D = D
        self.regimes: list[Regime] = list(regimes or ["symmetric", "asymmetric"])
        self.variants = list(variants or ["sym_psd", "sym_gen", "pure_asym",
                                          "full", "dual"])
        self.epochs = epochs
        self.with_decomposition = with_decomposition

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        per_seed: list[dict] = []
        for regime in self.regimes:
            for seed in range(self.n_seeds):
                seed_all(seed)
                data = make_pairwise_data(N=self.N, D=self.D, regime=regime,
                                          seed=seed)
                Z = data["Z"]
                R = data["R"]
                A_true = data["A_true"]
                A_S_true = (A_true + A_true.T) / 2
                A_A_true = (A_true - A_true.T) / 2
                rng = np.random.RandomState(seed + 100)
                mask_all = ~np.eye(self.N, dtype=bool)
                pair_idx = np.argwhere(mask_all)
                perm = rng.permutation(len(pair_idx))
                n_tr = len(pair_idx) // 2
                train_mask = np.zeros((self.N, self.N), dtype=bool)
                test_mask = np.zeros((self.N, self.N), dtype=bool)
                for k in perm[:n_tr]:
                    i, j = pair_idx[k]
                    train_mask[i, j] = True
                for k in perm[n_tr:]:
                    i, j = pair_idx[k]
                    test_mask[i, j] = True

                row: dict = {"regime": regime, "seed": seed, "variants": {}}
                if self.with_decomposition:
                    n2 = (A_true ** 2).sum() + 1e-12
                    row["true_sym_energy"] = float((A_S_true ** 2).sum() / n2)
                    row["true_asym_energy"] = float((A_A_true ** 2).sum() / n2)

                for v in self.variants:
                    torch.manual_seed(seed + 1000)
                    m = PairwisePredictor(d_in=self.D, variant=v)
                    train_pairwise(m, Z, R, train_mask, epochs=self.epochs)
                    te_mse = eval_pairwise(m, Z, R, test_mask)
                    entry: dict = {"test_mse": te_mse}
                    if self.with_decomposition:
                        with torch.no_grad():
                            A_learn = m.kernel_matrix().detach().cpu().numpy()
                        A_S_learn = (A_learn + A_learn.T) / 2
                        A_A_learn = (A_learn - A_learn.T) / 2
                        e_S, e_A = energy_split_np(A_learn)
                        entry.update({
                            "sym_energy_frac": e_S,
                            "asym_energy_frac": e_A,
                            "cos_sym": cosine_F_np(A_S_learn, A_S_true),
                            "cos_asym": cosine_F_np(A_A_learn, A_A_true),
                        })
                        if v == "dual":
                            with torch.no_grad():
                                M_S = m.M_S.detach().cpu().numpy()
                                M_A = m.M_A.detach().cpu().numpy()
                            entry["MS_norm"] = float(np.linalg.norm((M_S + M_S.T) / 2))
                            entry["MA_norm"] = float(np.linalg.norm((M_A - M_A.T) / 2))
                    row["variants"][v] = entry
                per_seed.append(row)

        metrics = self._aggregate(per_seed)
        report: AuditReport = {
            "config": {
                "n_seeds": self.n_seeds, "N": self.N, "D": self.D,
                "regimes": self.regimes, "variants": self.variants,
                "epochs": self.epochs,
                "with_decomposition": self.with_decomposition,
            },
            "metrics": metrics,
            "per_seed": per_seed,
        }
        if self.with_decomposition:
            report["decomposition"] = self._decomposition_summary(per_seed)
        if save_to is not None:
            save_report(report, save_to)
        return report

    def _aggregate(self, rows: list[dict]) -> dict:
        out: dict = {}
        for regime in self.regimes:
            ents = [r for r in rows if r["regime"] == regime]
            if not ents:
                continue
            summary: dict = {"variants": {}}
            for v in self.variants:
                vals = np.array([e["variants"][v]["test_mse"] for e in ents])
                summary["variants"][v] = {
                    "mean_test_mse": float(vals.mean()),
                    "std_test_mse": float(vals.std()),
                }
            out[regime] = summary
        return out

    def _decomposition_summary(self, rows: list[dict]) -> dict:
        out: dict = {}
        for regime in self.regimes:
            ents = [r for r in rows if r["regime"] == regime]
            if not ents:
                continue
            summary: dict = {
                "true_sym_energy": float(np.mean(
                    [e["true_sym_energy"] for e in ents])),
                "true_asym_energy": float(np.mean(
                    [e["true_asym_energy"] for e in ents])),
                "variants": {},
            }
            for v in self.variants:
                cos_sym = np.array([e["variants"][v]["cos_sym"] for e in ents])
                cos_asym = np.array([e["variants"][v]["cos_asym"] for e in ents])
                e_S = np.array([e["variants"][v]["sym_energy_frac"] for e in ents])
                e_A = np.array([e["variants"][v]["asym_energy_frac"] for e in ents])
                summary["variants"][v] = {
                    "cos_sym_mean": float(cos_sym.mean()),
                    "cos_asym_mean": float(cos_asym.mean()),
                    "sym_energy_frac_mean": float(e_S.mean()),
                    "asym_energy_frac_mean": float(e_A.mean()),
                }
            out[regime] = summary
        return out


__all__ = ["PairwiseAudit", "make_pairwise_data"]
