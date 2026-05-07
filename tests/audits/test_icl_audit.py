"""Smoke tests for the five ICL audits.

These exercise the public API on a tiny synthetic prior and verify the
:class:`AuditReport` shape; they do not attempt to reproduce the original
ARF-pretrained numbers (those are cached as JSON for chapter agents to load
directly).
"""
from __future__ import annotations

import json

from tabkernels.audits import (
    DualChannelAudit,
    PostHocDecomposition,
    ResidualBoost,
    StandalonePureAsym,
    ThreeWayAblation,
)

# Common smoke-mode kwargs: 1 seed, 5 train steps, batch of 2, n_q of 2.
_SMOKE = dict(n_seeds=1, n_steps=5, batch_size=2, n_q=2,
              d_x=4, d_model=8, n_heads=2, n_layers=1, dim_ff=8)


def test_three_way_ablation_smoke():
    report = ThreeWayAblation(**_SMOKE).run()
    assert "metrics" in report
    assert {"std", "sym", "psd"}.issubset(report["metrics"])
    row = report["per_seed"][0]
    for tag in ("std", "sym", "psd"):
        assert "holdout" in row[tag]


def test_three_way_is_deterministic():
    a = ThreeWayAblation(**_SMOKE).run()
    b = ThreeWayAblation(**_SMOKE).run()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_standalone_pure_asym_smoke():
    report = StandalonePureAsym(**_SMOKE).run()
    row = report["per_seed"][0]
    assert "pureasym" in row
    assert "holdout" in row["pureasym"]


def test_residual_boost_smoke():
    report = ResidualBoost(**_SMOKE).run()
    row = report["per_seed"][0]
    assert "alpha" in row
    assert "boost" in row
    assert "delta_holdout_mean" in report["metrics"]


def test_dual_channel_smoke():
    report = DualChannelAudit(**_SMOKE).run()
    row = report["per_seed"][0]
    for key in ("full", "sym_only", "asym_only"):
        assert "holdout" in row[key]


def test_post_hoc_decomposition_smoke():
    # The PyTorch built-in TransformerEncoder doesn't expose the raw
    # W_Q/W_K projections, but our StdAttnDecomposableTransformer does;
    # this audit uses the latter throughout.
    report = PostHocDecomposition(**_SMOKE).run()
    assert "decomposition" in report
    assert "sym_energy_frac_mean" in report["decomposition"]
    row = report["per_seed"][0]
    assert "energy_per_head" in row
    for mode in ("full", "sym", "asym"):
        assert "holdout" in row[mode]
