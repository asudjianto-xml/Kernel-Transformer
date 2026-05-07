"""Shared helpers for audit modules.

The abstract :class:`tabkernels.core.base.Audit` lives in ``core/base.py``;
this module adds the shared helpers (typed-dict report shape, JSON I/O,
seeding helpers) that all migrated audits use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch


class AuditReport(TypedDict, total=False):
    """Standard shape of an audit report.

    All migrated audits return a dict that is type-compatible with this; the
    fields that are present depend on the audit (see each subclass docstring).
    """

    config: dict
    metrics: dict
    per_seed: list[dict]
    decomposition: dict
    notes: str


def seed_all(seed: int) -> None:
    """Seed numpy + torch (CPU and CUDA) for deterministic audits."""
    # The legacy global RandomState is what the migrated audits use (and what
    # the cached JSON results reproduce against), so keep it seeded explicitly.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_python(obj: Any) -> Any:
    """Recursively convert torch tensors / numpy arrays to JSON-friendly values."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


def save_report(report: AuditReport, path: str | Path) -> None:
    """Save a report as JSON; tensors and numpy arrays are flattened to lists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(_to_python(report), f, indent=2)


def load_report(path: str | Path) -> AuditReport:
    """Load a previously-saved report from JSON."""
    with open(path) as f:
        return json.load(f)
