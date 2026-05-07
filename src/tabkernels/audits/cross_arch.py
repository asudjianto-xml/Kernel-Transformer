"""Cross-architecture kernel-lens audit (Chapter 18).

Given a dictionary of trained architectures (each exposing
``attention_blocks()``), apply the Chapter 13 post-hoc decomposition
diagnostic uniformly and aggregate the results into a single report.

The chapter uses this module to compare FT-Transformer, SAINT,
TabPFN-lite, and TabICL-lite on a common synthetic substrate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

from tabkernels.audits.base import AuditReport, save_report
from tabkernels.transparency import (
    decompose_attention,
    kernel_energy_split,
    recovery_cosine,
)


def _per_layer_energy(model: nn.Module) -> list[dict]:
    """Return [{layer, head, alpha_S, alpha_A}, ...] for one architecture."""
    out: list[dict] = []
    blocks = model.attention_blocks() if hasattr(model, "attention_blocks") else []
    for li, block in enumerate(blocks):
        # If the wrapper is itself an AttentionBlock with a non-default decompose,
        # use that. Otherwise drop into the inner StdAttentionDecomposable via
        # the transparency module.
        decomp = decompose_attention(block)
        if "B" not in decomp:
            continue
        B = decomp["B"]
        if B.dim() == 2:
            B = B.unsqueeze(0)
        for h in range(B.shape[0]):
            a_s, a_a = kernel_energy_split(B[h])
            out.append({"layer": li, "head": h,
                        "alpha_S": float(a_s), "alpha_A": float(a_a)})
    return out


def _per_layer_recovery(model: nn.Module,
                        target_B: torch.Tensor) -> list[dict]:
    """Per-layer / per-head Frobenius cosine of (B_S, B_A) to a target B."""
    rows: list[dict] = []
    blocks = model.attention_blocks() if hasattr(model, "attention_blocks") else []
    for li, block in enumerate(blocks):
        decomp = decompose_attention(block)
        if "B" not in decomp:
            continue
        B = decomp["B"]
        if B.dim() == 2:
            B = B.unsqueeze(0)
        for h in range(B.shape[0]):
            cos_sym, cos_asym = recovery_cosine(B[h], target_B)
            rows.append({"layer": li, "head": h,
                         "cos_sym": float(cos_sym),
                         "cos_asym": float(cos_asym)})
    return rows


def run_cross_arch_audit(
    architectures: dict[str, nn.Module],
    target_B: Optional[torch.Tensor] = None,
    save_to: Optional[str | Path] = None,
) -> AuditReport:
    """Apply the Chapter 13 diagnostic across multiple trained architectures.

    Parameters
    ----------
    architectures : dict[str, nn.Module]
        Map from architecture name (used as a row label in tables) to the
        trained model. Each model is expected to expose ``attention_blocks()``.
    target_B : Tensor or None
        Optional ``(d, d)`` reference bilinear form against which we report
        the Frobenius cosine of every attention block's symmetric part.
        ``None`` skips the recovery section of the report.
    save_to : path or None
        If given, also persist the report as JSON via
        :func:`tabkernels.audits.base.save_report`.
    """
    per_arch: dict[str, dict] = {}
    summary_rows: list[dict] = []
    for name, model in architectures.items():
        energies = _per_layer_energy(model)
        if energies:
            mean_alpha_s = sum(e["alpha_S"] for e in energies) / len(energies)
            mean_alpha_a = sum(e["alpha_A"] for e in energies) / len(energies)
        else:
            mean_alpha_s = float("nan"); mean_alpha_a = float("nan")
        recovery = _per_layer_recovery(model, target_B) if target_B is not None else []
        if recovery:
            mean_cos = sum(r["cos_sym"] for r in recovery) / len(recovery)
        else:
            mean_cos = float("nan")
        per_arch[name] = {
            "energies": energies,
            "recovery": recovery,
            "n_blocks": len(energies),
            "mean_alpha_S": mean_alpha_s,
            "mean_alpha_A": mean_alpha_a,
            "mean_cos_sym": mean_cos,
        }
        summary_rows.append({
            "architecture": name,
            "n_blocks": len(energies),
            "mean_alpha_S": mean_alpha_s,
            "mean_alpha_A": mean_alpha_a,
            "mean_cos_sym": mean_cos,
        })

    report: AuditReport = {
        "config": {
            "kind": "cross_arch_audit",
            "architectures": list(architectures.keys()),
            "has_target": target_B is not None,
        },
        "metrics": {"summary": summary_rows},
        "per_seed": [{"architecture": k, **v} for k, v in per_arch.items()],
    }
    if save_to is not None:
        save_report(report, save_to)
    return report


__all__ = ["run_cross_arch_audit"]
