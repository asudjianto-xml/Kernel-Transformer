"""Smoke + regression tests for :class:`tabkernels.audits.PairwiseAudit`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tabkernels.audits import PairwiseAudit
from tabkernels.audits.pairwise import make_pairwise_data

CACHED = Path(
    "/home/asudjianto/jupyterlab/similarity-hierarchy-research/affinity/book/"
    "data/cached_audits/pairwise_test_mse.json"
)
CACHED_DECOMP = Path(
    "/home/asudjianto/jupyterlab/similarity-hierarchy-research/affinity/book/"
    "data/cached_audits/pairwise_decomposition.json"
)


def test_make_pairwise_data_shapes():
    d = make_pairwise_data(N=15, D=4, regime="asymmetric", seed=0)
    assert d["Z"].shape == (15, 4)
    assert d["R"].shape == (15, 15)
    assert d["A_true"].shape == (4, 4)


def test_smoke_runs():
    audit = PairwiseAudit(
        n_seeds=1, N=20, D=4,
        regimes=["asymmetric"],
        variants=["sym_psd", "full"],
        epochs=20,
    )
    report = audit.run()
    assert "metrics" in report
    assert "decomposition" in report
    assert len(report["per_seed"]) == 1
    row = report["per_seed"][0]
    assert "test_mse" in row["variants"]["sym_psd"]
    assert "cos_sym" in row["variants"]["full"]


def test_smoke_is_deterministic():
    a = PairwiseAudit(n_seeds=1, N=20, D=4,
                      regimes=["asymmetric"], variants=["sym_psd"],
                      epochs=20).run()
    b = PairwiseAudit(n_seeds=1, N=20, D=4,
                      regimes=["asymmetric"], variants=["sym_psd"],
                      epochs=20).run()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_smoke_without_decomposition():
    audit = PairwiseAudit(
        n_seeds=1, N=20, D=4,
        regimes=["asymmetric"], variants=["sym_psd"],
        epochs=20, with_decomposition=False,
    )
    report = audit.run()
    assert "decomposition" not in report
    row = report["per_seed"][0]
    # No diagnostic keys when decomposition is off.
    assert set(row["variants"]["sym_psd"]) == {"test_mse"}


def test_regression_against_cached_test_mse():
    """In the asymmetric regime ``full`` should beat ``sym_psd`` on test MSE
    (this is the headline of the cached audit)."""
    if not CACHED.exists():
        return
    cached = json.loads(CACHED.read_text())
    asym = [r for r in cached if r["regime"] == "asymmetric"]
    assert len(asym) >= 1
    full = np.array([r["variants"]["full"] for r in asym])
    sym = np.array([r["variants"]["sym_psd"] for r in asym])
    assert full.mean() < sym.mean()


def test_regression_against_cached_decomposition():
    """``full`` recovers both sym and asym components on the asym regime."""
    if not CACHED_DECOMP.exists():
        return
    cached = json.loads(CACHED_DECOMP.read_text())
    asym = [r for r in cached if r["regime"] == "asymmetric"]
    assert len(asym) >= 1
    cos_sym = np.array([r["variants"]["full"]["cos_sym"] for r in asym])
    cos_asym = np.array([r["variants"]["full"]["cos_asym"] for r in asym])
    assert cos_sym.mean() > 0.5
    assert cos_asym.mean() > 0.5
