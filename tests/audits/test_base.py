"""Smoke tests for the audit-base helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tabkernels.audits.base import (
    AuditReport,
    _to_python,
    load_report,
    save_report,
    seed_all,
)


def test_seed_all_is_deterministic():
    seed_all(42)
    a1 = torch.randn(4)
    b1 = np.random.randn(4)  # noqa: NPY002 -- legacy state is what audits seed
    seed_all(42)
    a2 = torch.randn(4)
    b2 = np.random.randn(4)  # noqa: NPY002
    assert torch.allclose(a1, a2)
    assert np.allclose(b1, b2)


def test_to_python_handles_tensors_and_arrays():
    obj = {
        "t": torch.tensor([1.0, 2.0]),
        "a": np.array([3, 4]),
        "nested": [torch.tensor([5.0]), {"x": np.float32(6.0)}],
    }
    out = _to_python(obj)
    assert out["t"] == [1.0, 2.0]
    assert out["a"] == [3, 4]
    assert out["nested"][0] == [5.0]
    assert out["nested"][1]["x"] == 6.0


def test_save_load_roundtrip(tmp_path: Path):
    report: AuditReport = {
        "config": {"foo": 1},
        "metrics": {"accuracy": 0.5},
        "per_seed": [{"seed": 0, "tensor": torch.zeros(2)}],
    }
    p = tmp_path / "r.json"
    save_report(report, p)
    loaded = load_report(p)
    assert loaded["config"] == {"foo": 1}
    assert loaded["metrics"] == {"accuracy": 0.5}
    assert loaded["per_seed"][0]["tensor"] == [0.0, 0.0]
    # Plain JSON-readable.
    assert json.loads(p.read_text())["config"]["foo"] == 1
