"""ICL audit (Chapter 10 §10.5).

Migrated from five source scripts:

* :class:`ThreeWayAblation`        — ``symmetric_kernel_icl_arf.py`` (3-way only)
* :class:`StandalonePureAsym`      — ``icl_pure_asym_scratch.py``
* :class:`ResidualBoost`           — ``icl_pure_asym_boost.py``
* :class:`DualChannelAudit`        — ``icl_dual_channel.py``
* :class:`PostHocDecomposition`    — ``icl_post_hoc_decomp.py``

The original scripts depend on a heavyweight ARF prior (``corpus/`` of trained
forests) and several thousand pretraining steps per seed. For module
reproducibility we expose a ``prior`` argument; the default uses the synthetic
Gaussian-feature + linear-threshold prior in :func:`._helpers.make_synthetic_icl_episode`,
which lets the smoke tests run in seconds. To regenerate the cached JSON
results, callers must wire in the original ARF prior via ``train_fn`` /
``evaluate_fn``.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from tabkernels.core.base import Audit

from ._helpers import (
    evaluate_icl_synthetic,
    get_device,
    train_icl_synthetic,
)
from ._models import (
    BoostedSymPureAsym,
    DualSymAsymICLTransformer,
    PSDICLTransformer,
    PureAsymICLTransformer,
    StdAttnDecomposableTransformer,
    StdICLTransformer,
    SymSoftmaxICLTransformer,
    measure_kernel_decomposition,
    set_score_mode,
)
from .base import AuditReport, save_report, seed_all

# Default architecture knobs match the originals.
_DEFAULTS = dict(
    d_x=8, d_model=64, n_heads=4, n_layers=1, dim_ff=128, dropout=0.0,
    n_classes=2,
)


def _build(cls: type, **overrides) -> torch.nn.Module:
    """Instantiate one of the ICL transformer wrappers and move to device."""
    kw = {**_DEFAULTS, **overrides}
    model = cls(**kw).to(get_device())
    return model


# ----------------------------- ThreeWayAblation -----------------------------


class ThreeWayAblation(Audit):
    """3-way ablation: ``Std`` vs ``SymSoftmax`` vs ``PSD-NW`` (§10.5.1).

    Trains each architecture on the prior, evaluates held-out + label-swap +
    context-size scaling, and reports both raw means and the
    ``(Δ_sym, Δ_norm, Δ_total)`` decomposition that drives the chapter's
    decomposition figure.
    """

    def __init__(self, *, n_seeds: int = 3, n_steps: int = 200,
                 batch_size: int = 16, n_q: int = 8,
                 train_fn: Callable | None = None,
                 evaluate_fn: Callable | None = None,
                 **arch_overrides) -> None:
        self.n_seeds = n_seeds
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_q = n_q
        self.arch_overrides = arch_overrides
        self.train_fn = train_fn or train_icl_synthetic
        self.evaluate_fn = evaluate_fn or evaluate_icl_synthetic

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        per_seed: list[dict] = []
        for seed in range(self.n_seeds):
            seed_row: dict = {"seed": seed}
            for tag, cls in (
                ("std", StdICLTransformer),
                ("sym", SymSoftmaxICLTransformer),
                ("psd", PSDICLTransformer),
            ):
                seed_all(seed)
                m = _build(cls, **self.arch_overrides)
                self.train_fn(m, n_steps=self.n_steps,
                              batch_size=self.batch_size,
                              n_q=self.n_q, seed=seed)
                ev = self.evaluate_fn(m, seed=seed * 1000 + 7919)
                seed_row[tag] = ev
                del m
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            per_seed.append(seed_row)

        report: AuditReport = {
            "config": {
                "n_seeds": self.n_seeds, "n_steps": self.n_steps,
                "batch_size": self.batch_size, "n_q": self.n_q,
                "arch_overrides": self.arch_overrides,
            },
            "metrics": self._aggregate(per_seed),
            "per_seed": per_seed,
            "notes": (
                "Default smoke-mode uses the synthetic prior in _helpers; the "
                "cached JSON was generated against the ARF corpus."
            ),
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        out: dict = {}
        for tag in ("std", "sym", "psd"):
            holdout = np.array([r[tag]["holdout"] for r in rows])
            swap = np.array([r[tag]["swap"] for r in rows])
            out[tag] = {
                "holdout_mean": float(holdout.mean()),
                "holdout_std": float(holdout.std()),
                "swap_mean": float(swap.mean()),
            }
        std = np.array([r["std"]["holdout"] for r in rows])
        sym = np.array([r["sym"]["holdout"] for r in rows])
        psd = np.array([r["psd"]["holdout"] for r in rows])
        out["delta"] = {
            "delta_sym": float((sym - std).mean()),
            "delta_norm": float((psd - sym).mean()),
            "delta_total": float((psd - std).mean()),
        }
        return out


# ----------------------------- StandalonePureAsym -----------------------------


class StandalonePureAsym(Audit):
    """Train ``Std``, ``Sym``, ``PureAsym`` from scratch (§10.5.2)."""

    def __init__(self, *, n_seeds: int = 3, n_steps: int = 200,
                 batch_size: int = 16, n_q: int = 8,
                 train_fn: Callable | None = None,
                 evaluate_fn: Callable | None = None,
                 **arch_overrides) -> None:
        self.n_seeds = n_seeds
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_q = n_q
        self.arch_overrides = arch_overrides
        self.train_fn = train_fn or train_icl_synthetic
        self.evaluate_fn = evaluate_fn or evaluate_icl_synthetic

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        per_seed: list[dict] = []
        for seed in range(self.n_seeds):
            seed_row: dict = {"seed": seed}
            for tag, cls in (
                ("std",      StdICLTransformer),
                ("sym",      SymSoftmaxICLTransformer),
                ("pureasym", PureAsymICLTransformer),
            ):
                seed_all(seed)
                m = _build(cls, **self.arch_overrides)
                self.train_fn(m, n_steps=self.n_steps,
                              batch_size=self.batch_size,
                              n_q=self.n_q, seed=seed)
                ev = self.evaluate_fn(m, seed=seed * 1000 + 7919)
                seed_row[tag] = ev
                del m
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            per_seed.append(seed_row)
        report: AuditReport = {
            "config": {
                "n_seeds": self.n_seeds, "n_steps": self.n_steps,
                "batch_size": self.batch_size, "n_q": self.n_q,
                "arch_overrides": self.arch_overrides,
            },
            "metrics": self._aggregate(per_seed),
            "per_seed": per_seed,
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        out: dict = {}
        for tag in ("std", "sym", "pureasym"):
            holdout = np.array([r[tag]["holdout"] for r in rows])
            out[tag] = {
                "holdout_mean": float(holdout.mean()),
                "holdout_std": float(holdout.std()),
            }
        return out


# ----------------------------- ResidualBoost -----------------------------


class ResidualBoost(Audit):
    """``Sym`` (frozen) + ``alpha * PureAsymDelta`` (§10.5.3)."""

    def __init__(self, *, n_seeds: int = 3, n_steps: int = 200,
                 batch_size: int = 16, n_q: int = 8,
                 train_fn: Callable | None = None,
                 evaluate_fn: Callable | None = None,
                 **arch_overrides) -> None:
        self.n_seeds = n_seeds
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_q = n_q
        self.arch_overrides = arch_overrides
        self.train_fn = train_fn or train_icl_synthetic
        self.evaluate_fn = evaluate_fn or evaluate_icl_synthetic

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        per_seed: list[dict] = []
        for seed in range(self.n_seeds):
            seed_all(seed)
            sym = _build(SymSoftmaxICLTransformer, **self.arch_overrides)
            self.train_fn(sym, n_steps=self.n_steps,
                          batch_size=self.batch_size,
                          n_q=self.n_q, seed=seed)
            sym_eval = self.evaluate_fn(sym, seed=seed * 1000 + 7919)

            seed_all(seed)
            boost = BoostedSymPureAsym(
                sym, **{**_DEFAULTS, **self.arch_overrides}
            ).to(get_device())
            self.train_fn(boost, n_steps=self.n_steps,
                          batch_size=self.batch_size,
                          n_q=self.n_q, seed=seed)
            boost_eval = self.evaluate_fn(boost, seed=seed * 1000 + 7919)
            alpha = float(boost.alpha.item())
            per_seed.append({
                "seed": seed, "alpha": alpha,
                "sym": sym_eval, "boost": boost_eval,
            })
            del sym, boost
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report: AuditReport = {
            "config": {
                "n_seeds": self.n_seeds, "n_steps": self.n_steps,
                "batch_size": self.batch_size, "n_q": self.n_q,
                "arch_overrides": self.arch_overrides,
            },
            "metrics": self._aggregate(per_seed),
            "per_seed": per_seed,
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        sym = np.array([r["sym"]["holdout"] for r in rows])
        boost = np.array([r["boost"]["holdout"] for r in rows])
        alphas = np.array([r["alpha"] for r in rows])
        return {
            "sym_holdout_mean": float(sym.mean()),
            "boost_holdout_mean": float(boost.mean()),
            "delta_holdout_mean": float((boost - sym).mean()),
            "alpha_mean": float(alphas.mean()),
            "alpha_std": float(alphas.std()),
        }


# ----------------------------- DualChannelAudit -----------------------------


class DualChannelAudit(Audit):
    """Joint Sym + PureAsym branches with co-training (§10.5.4)."""

    def __init__(self, *, n_seeds: int = 3, n_steps: int = 200,
                 batch_size: int = 16, n_q: int = 8,
                 train_fn: Callable | None = None,
                 evaluate_fn: Callable | None = None,
                 **arch_overrides) -> None:
        self.n_seeds = n_seeds
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_q = n_q
        self.arch_overrides = arch_overrides
        self.train_fn = train_fn or train_icl_synthetic
        self.evaluate_fn = evaluate_fn or evaluate_icl_synthetic

    def _branch_eval(self, model: torch.nn.Module, branch: str,
                     seed: int) -> dict:
        """Evaluate using only one branch's logits."""
        orig = model.forward

        def patched(X_ctx, y_ctx, X_q, return_branches=False):
            out, sym_l, asym_l = orig(X_ctx, y_ctx, X_q, return_branches=True)
            z = sym_l if branch == "sym" else asym_l
            return z if not return_branches else (z, sym_l, asym_l)

        model.forward = patched
        try:
            ev = self.evaluate_fn(model, seed=seed)
        finally:
            model.forward = orig
        return ev

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        per_seed: list[dict] = []
        for seed in range(self.n_seeds):
            seed_all(seed)
            m = _build(DualSymAsymICLTransformer, **self.arch_overrides)
            self.train_fn(m, n_steps=self.n_steps,
                          batch_size=self.batch_size,
                          n_q=self.n_q, seed=seed)
            full = self.evaluate_fn(m, seed=seed * 1000 + 7919)
            sym_only = self._branch_eval(m, "sym", seed=seed * 1000 + 7919)
            asym_only = self._branch_eval(m, "asym", seed=seed * 1000 + 7919)
            per_seed.append({
                "seed": seed, "full": full,
                "sym_only": sym_only, "asym_only": asym_only,
            })
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report: AuditReport = {
            "config": {
                "n_seeds": self.n_seeds, "n_steps": self.n_steps,
                "batch_size": self.batch_size, "n_q": self.n_q,
                "arch_overrides": self.arch_overrides,
            },
            "metrics": self._aggregate(per_seed),
            "per_seed": per_seed,
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        out: dict = {}
        for key in ("full", "sym_only", "asym_only"):
            vals = np.array([r[key]["holdout"] for r in rows])
            out[key] = {
                "holdout_mean": float(vals.mean()),
                "holdout_std": float(vals.std()),
            }
        return out


# ----------------------------- PostHocDecomposition -----------------------------


class PostHocDecomposition(Audit):
    """Train Std-Attn freely; project sym-only / asym-only at inference (§10.5.5)."""

    def __init__(self, *, n_seeds: int = 3, n_steps: int = 200,
                 batch_size: int = 16, n_q: int = 8,
                 train_fn: Callable | None = None,
                 evaluate_fn: Callable | None = None,
                 **arch_overrides) -> None:
        self.n_seeds = n_seeds
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_q = n_q
        self.arch_overrides = arch_overrides
        self.train_fn = train_fn or train_icl_synthetic
        self.evaluate_fn = evaluate_fn or evaluate_icl_synthetic

    def run(self, *, save_to: str | Path | None = None) -> AuditReport:  # type: ignore[override]
        per_seed: list[dict] = []
        for seed in range(self.n_seeds):
            seed_all(seed)
            m = _build(StdAttnDecomposableTransformer, **self.arch_overrides)
            set_score_mode(m, "full")
            self.train_fn(m, n_steps=self.n_steps,
                          batch_size=self.batch_size,
                          n_q=self.n_q, seed=seed)
            energies = measure_kernel_decomposition(m)

            results: dict[str, dict] = {}
            for mode in ("full", "sym", "asym"):
                set_score_mode(m, mode)
                results[mode] = self.evaluate_fn(m, seed=seed * 1000 + 7919)
            per_seed.append({
                "seed": seed,
                "energy_per_head": energies,
                **results,
            })
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report: AuditReport = {
            "config": {
                "n_seeds": self.n_seeds, "n_steps": self.n_steps,
                "batch_size": self.batch_size, "n_q": self.n_q,
                "arch_overrides": self.arch_overrides,
            },
            "metrics": self._aggregate(per_seed),
            "per_seed": per_seed,
            "decomposition": self._decomposition_summary(per_seed),
        }
        if save_to is not None:
            save_report(report, save_to)
        return report

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        out: dict = {}
        for mode in ("full", "sym", "asym"):
            vals = np.array([r[mode]["holdout"] for r in rows])
            out[mode] = {
                "holdout_mean": float(vals.mean()),
                "holdout_std": float(vals.std()),
            }
        return out

    @staticmethod
    def _decomposition_summary(rows: list[dict]) -> dict:
        all_energies = []
        for r in rows:
            all_energies.extend(r["energy_per_head"])
        if not all_energies:
            return {"sym_energy_frac_mean": 0.0, "asym_energy_frac_mean": 0.0}
        arr = np.array(all_energies)
        return {
            "sym_energy_frac_mean": float(arr[:, 0].mean()),
            "asym_energy_frac_mean": float(arr[:, 1].mean()),
        }


__all__ = [
    "ThreeWayAblation",
    "StandalonePureAsym",
    "ResidualBoost",
    "DualChannelAudit",
    "PostHocDecomposition",
]
