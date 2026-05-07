"""Smoke + regression tests for :mod:`tabkernels.audits.benchmark`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabkernels.audits import (
    Corner,
    aggregate_winrates,
    plot_per_corner_winners,
    run_360cell_benchmark,
)
from tabkernels.audits.benchmark import (
    ASYMMETRIC_VARIANTS,
    SYMMETRIC_VARIANTS,
    _headtohead_per_cell,
)

CACHED = Path(
    "/home/asudjianto/jupyterlab/similarity-hierarchy-research/affinity/book/"
    "data/cached_audits/benchmark_360cell.json"
)


# --------------------------------------------------------------------------- #
# Smoke: 2 datasets x 2 corners x 1 seed slice runs and aggregates.           #
# --------------------------------------------------------------------------- #


def _toy_datasets() -> list[dict]:
    return [
        dict(name="ToyReg", task="regression", n=64, p=4, metric="mse"),
        dict(name="ToyCla", task="classification", n=64, p=4, metric="ce"),
    ]


def _toy_corners() -> list[Corner]:
    return [
        Corner(head_mode="nw", depth=1, task="regression"),
        Corner(head_mode="mlp", depth=1, task="regression"),
        Corner(head_mode="nw", depth=1, task="classification"),
        Corner(head_mode="mlp", depth=1, task="classification"),
    ]


def test_run_360cell_benchmark_slice_only_smoke():
    """End-to-end smoke: slice runs, report has expected shape."""
    report = run_360cell_benchmark(
        datasets=_toy_datasets(),
        variants=list(ASYMMETRIC_VARIANTS) + list(SYMMETRIC_VARIANTS),
        corners=_toy_corners(),
        seeds=1,
        head_counts=(1, 4),
        slice_only=True,
    )
    assert "config" in report
    assert "per_seed" in report
    assert "metrics" in report
    assert report["config"]["slice_only"] is True
    # 2 datasets * 4 corners (filtered by task = 1 dataset per corner)
    # * 2 H * 7 variants * 1 seed = 4 corners * 1 dataset * 2H * 7v = 56
    assert len(report["per_seed"]) == 56


def test_run_360cell_benchmark_requires_train_fn_when_not_slice():
    with pytest.raises(ValueError, match="explicit train_fn"):
        run_360cell_benchmark(
            datasets=_toy_datasets(),
            variants=list(ASYMMETRIC_VARIANTS),
            corners=_toy_corners(),
            seeds=1,
            slice_only=False,
        )


def test_aggregate_winrates_schema():
    """Aggregator emits the per-corner DataFrame with the expected schema."""
    report = run_360cell_benchmark(
        datasets=_toy_datasets(),
        variants=list(ASYMMETRIC_VARIANTS) + list(SYMMETRIC_VARIANTS),
        corners=_toy_corners(),
        seeds=1,
        head_counts=(1, 4),
        slice_only=True,
    )
    df = aggregate_winrates(report)
    expected_cols = {
        "head_mode", "depth", "task", "n_cells",
        "sym_win_frac", "median_delta",
        "std_attn_wins", "best_sym_wins", "best_sym_variant",
    }
    assert expected_cols.issubset(df.columns)
    # 4 (head_mode x task) per-corner rows + 2 head-mode roll-ups + 1 ALL
    assert len(df) == 4 + 2 + 1
    # The slice biases symmetric variants down, so sym_win_frac should be 1.0
    # on every per-corner row.
    per_corner = df[df["depth"] != "any"]
    assert (per_corner["sym_win_frac"] == 1.0).all()


def test_plot_per_corner_winners_returns_figure():
    report = run_360cell_benchmark(
        datasets=_toy_datasets(),
        variants=list(ASYMMETRIC_VARIANTS) + list(SYMMETRIC_VARIANTS),
        corners=_toy_corners(),
        seeds=1,
        head_counts=(1, 4),
        slice_only=True,
    )
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_per_corner_winners(report)
    assert fig is not None
    assert len(fig.axes) == 1


def test_aggregate_rejects_empty_report():
    with pytest.raises(ValueError, match="no cells"):
        aggregate_winrates({"per_seed": []})


# --------------------------------------------------------------------------- #
# Regression: cached JSON reproduces the §6 headline rates.                   #
# --------------------------------------------------------------------------- #


def test_cached_headline_rates_match_paper():
    """78% / 88% / 69% from the cached JSON, per flagship paper §6."""
    if not CACHED.exists():
        pytest.skip("Cached benchmark JSON not present.")
    cached = json.loads(CACHED.read_text())
    metrics = cached["metrics"]
    assert metrics["headline_overall_n"] == 360
    assert metrics["headline_overall_sym_win_frac"] == pytest.approx(0.7833, abs=1e-3)
    assert metrics["by_head_mode"]["mlp"]["sym_win_frac"] == pytest.approx(
        0.8778, abs=1e-3
    )
    assert metrics["by_head_mode"]["nw"]["sym_win_frac"] == pytest.approx(
        0.6889, abs=1e-3
    )


def test_cached_per_corner_winner_counts_match_paper():
    """Diff-Cheb totals 100 wins, Std-Attn 65, PSD-NW 57 across 360 cells."""
    if not CACHED.exists():
        pytest.skip("Cached benchmark JSON not present.")
    cached = json.loads(CACHED.read_text())
    pcc = cached["metrics"]["per_corner_winner_counts"]
    totals: dict[str, int] = {}
    for _, payload in pcc.items():
        for var, c in payload["counts"].items():
            totals[var] = totals.get(var, 0) + c
    assert totals["diff_cheb"] == 100
    assert totals["std_attn"] == 65
    assert totals["psd_nw"] == 57


def test_headtohead_helper_handles_missing_variants():
    """If a cell is missing both Std-Attn and any symmetric variant it is
    silently dropped, not raised."""
    cells = [
        dict(variant="aff_nw", dataset="X", task="regression",
             H=1, n_layers=1, head_mode="nw", mean=0.5, std=0.0,
             raw=[], mean_acc=None),
    ]
    assert _headtohead_per_cell(cells) == []
