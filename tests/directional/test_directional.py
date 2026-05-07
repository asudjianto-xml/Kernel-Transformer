"""Tests for tabkernels.directional (Chapter 25)."""
from __future__ import annotations

import numpy as np
import pytest

from tabkernels.directional import (
    ARGenerator,
    DAGGenerator,
    causal_kernel_inference,
    make_treatment_effect_data,
)


def test_dag_generator_returns_consistent_shapes():
    g = DAGGenerator(D=4)
    data = g.sample(N=50, seed=0)
    assert data["X"].shape == (50, 4)
    assert data["y"].shape == (50,)
    assert len(data["parents"]) == 50
    assert data["kind"] == "dag"


def test_dag_generator_seed_reproducible():
    g = DAGGenerator(D=4)
    a = g.sample(N=20, seed=7)
    b = g.sample(N=20, seed=7)
    assert np.allclose(a["X"], b["X"])
    assert np.allclose(a["y"], b["y"])


def test_dag_oracle_runs():
    g = DAGGenerator(D=4)
    data = g.sample(N=50, seed=0)
    idx_tr = np.arange(0, 30)
    idx_te = np.arange(30, 50)
    pred = g.oracle(data, idx_tr, idx_te)
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()


def test_ar_generator_shapes_and_kind():
    g = ARGenerator(D=4, p=2)
    data = g.sample(T=100, seed=1)
    assert data["X"].shape == (100, 4)
    assert data["y"].shape == (100,)
    assert data["kind"] == "ar"
    assert data["p"] == 2


def test_ar_oracle_predicts_finite():
    g = ARGenerator(D=4, p=2)
    data = g.sample(T=80, seed=0)
    idx_tr = np.arange(0, 40)
    idx_te = np.arange(40, 80)
    pred = g.oracle(data, idx_tr, idx_te)
    assert pred.shape == (40,)
    assert np.isfinite(pred).all()


def test_treatment_data_balanced_arms():
    data = make_treatment_effect_data(N=400, d=4, seed=0)
    n_treat = (data["A"] == 1).sum()
    n_ctrl = (data["A"] == 0).sum()
    # both arms should have at least 10% of the data.
    assert n_treat >= 40 and n_ctrl >= 40


def test_treatment_data_has_true_ate():
    data = make_treatment_effect_data(N=200, d=4, tau_strength=1.0, seed=0)
    assert "true_ate" in data and isinstance(data["true_ate"], float)
    assert "tau" in data and data["tau"].shape == (200,)


def test_causal_kernel_recovers_positive_ate():
    """With tau_strength=1.0 the true ATE is positive; estimator should agree."""
    data = make_treatment_effect_data(N=400, d=4, tau_strength=1.0, seed=0)
    out = causal_kernel_inference(data, sigma=1.0)
    assert out["ate_hat"] > 0
    # sanity: estimator should be within a factor of 3 of the truth.
    assert abs(out["ate_hat"] - out["true_ate"]) < 3 * abs(out["true_ate"])


def test_causal_kernel_returns_cate_array():
    data = make_treatment_effect_data(N=200, d=4, seed=0)
    out = causal_kernel_inference(data, sigma=1.0)
    assert out["cate_hat"].shape == (200,)
    assert np.isfinite(out["cate_hat"]).all()


def test_causal_kernel_raises_on_empty_arm():
    """If one arm is empty, the estimator must error."""
    data = make_treatment_effect_data(N=100, d=4, seed=0)
    data["A"] = np.zeros_like(data["A"])  # everyone in control arm
    with pytest.raises(ValueError):
        causal_kernel_inference(data)
