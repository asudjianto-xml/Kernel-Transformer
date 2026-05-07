"""Smoke tests for :class:`tabkernels.audits.StructureAnalysis`."""
from __future__ import annotations

import json

from tabkernels.audits import StructureAnalysis


def test_smoke_runs():
    audit = StructureAnalysis(
        N_total=50, N_train=25, D=4, n_seeds=1,
        structures=["sym"], tasks=["reg"],
        d_emb=4, epochs=20,
    )
    report = audit.run()
    assert "metrics" in report
    assert "decomposition" in report
    assert len(report["per_seed"]) == 1
    row = report["per_seed"][0]
    assert "B_sym_frac" in row["std"]
    assert "M_S_norm" in row["dual"]
    assert "sym_frac" in row["oracle_sample_gram"]


def test_smoke_is_deterministic():
    a = StructureAnalysis(
        N_total=50, N_train=25, D=4, n_seeds=1,
        structures=["sym"], tasks=["reg"],
        d_emb=4, epochs=20,
    ).run()
    b = StructureAnalysis(
        N_total=50, N_train=25, D=4, n_seeds=1,
        structures=["sym"], tasks=["reg"],
        d_emb=4, epochs=20,
    ).run()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
