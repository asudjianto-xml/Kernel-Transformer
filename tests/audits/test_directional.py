"""Smoke tests for the directional audits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tabkernels.audits import DirectionalAudit, TimeOrderedAuditAR
from tabkernels.audits.directional import (
    make_ar_data,
    make_dag_data,
    oracle_predict_ar,
    oracle_predict_dag,
)

CACHED_RANDOM = Path(
    "/home/asudjianto/jupyterlab/similarity-hierarchy-research/affinity/book/"
    "data/cached_audits/directional_random.json"
)


def test_make_dag_data_shapes():
    d = make_dag_data(N=20, D=4, seed=0)
    assert d["X"].shape == (20, 4)
    assert d["y"].shape == (20,)
    assert d["order"].shape == (20,)


def test_make_ar_data_shapes():
    d = make_ar_data(T=30, D=4, p=2, seed=0)
    assert d["X"].shape == (30, 4)
    assert d["y"].shape == (30,)
    assert d["order"].shape == (30,)


def test_oracle_predicts_have_right_shape():
    d = make_dag_data(N=20, D=4, seed=0)
    idx_tr = np.arange(0, 10)
    idx_te = np.arange(10, 20)
    p = oracle_predict_dag(d, idx_tr, idx_te)
    assert p.shape == (10,)
    d2 = make_ar_data(T=30, D=4, p=2, seed=0)
    p2 = oracle_predict_ar(d2, np.arange(15), np.arange(15, 30))
    assert p2.shape == (15,)


def test_directional_smoke():
    audit = DirectionalAudit(n_seeds=1, N_dag=30, T_ar=30, D=4,
                             d_emb=4, epochs=20)
    report = audit.run()
    assert isinstance(report["per_seed"], list)
    assert len(report["per_seed"]) == 2  # one DAG + one AR
    assert "metrics" in report


def test_directional_is_deterministic():
    a = DirectionalAudit(n_seeds=1, N_dag=30, T_ar=30, D=4,
                         d_emb=4, epochs=20).run()
    b = DirectionalAudit(n_seeds=1, N_dag=30, T_ar=30, D=4,
                         d_emb=4, epochs=20).run()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_time_ordered_smoke():
    audit = TimeOrderedAuditAR(n_seeds=1, T=20, D=4, p=2,
                               d_emb=4, epochs=20)
    report = audit.run()
    assert len(report["per_seed"]) == 1
    row = report["per_seed"][0]
    for tag in ("std", "sym_psd", "std_mask", "sym_mask"):
        assert "batch_mse" in row["variants"][tag]
        assert "ar_mse" in row["variants"][tag]


def test_regression_directional_random():
    """Cached MSEs are non-negative finite numbers."""
    if not CACHED_RANDOM.exists():
        return
    cached = json.loads(CACHED_RANDOM.read_text())
    for row in cached:
        for val in row["variants"].values():
            assert val >= 0
            assert val < 1e6
