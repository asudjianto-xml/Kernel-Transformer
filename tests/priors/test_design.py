"""Tests for tabkernels.priors.design (Chapter 22)."""
from __future__ import annotations

import pytest
import torch

from tabkernels.core.base import Prior
from tabkernels.priors import (
    KNOWN_ARCHS,
    KNOWN_TASKS,
    MLPSCMPrior,
    SCMPrior,
    recommend_prior,
)


def test_returns_prior_for_every_known_arch():
    for arch in KNOWN_ARCHS:
        p = recommend_prior(architecture=arch, task="regression")
        assert isinstance(p, Prior)


def test_symmetric_archs_get_hybrid_prior():
    p = recommend_prior(architecture="sym_psd", task="regression")
    assert isinstance(p, MLPSCMPrior)


def test_asymmetric_archs_get_scm_prior():
    p = recommend_prior(architecture="dual", task="regression")
    assert isinstance(p, SCMPrior)


def test_directional_task_overrides_arch():
    """Directional task should return an SCM prior even for a symmetric arch."""
    p = recommend_prior(architecture="sym_psd", task="directional")
    assert isinstance(p, SCMPrior)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        recommend_prior(architecture="not_a_real_arch", task="regression")


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        recommend_prior(architecture="std", task="not_a_real_task")


def test_recommended_prior_emits_valid_episodes():
    for arch in ["sym_psd", "dual"]:
        p = recommend_prior(architecture=arch, task="regression", seed=0)
        X_c, y_c, X_q, y_q = p.sample_episode(8, 4, d=4, seed=42)
        assert X_c.shape == (8, 4)
        assert y_c.shape == (8,)
        assert torch.isfinite(X_c).all() and torch.isfinite(y_c).all()


def test_seed_for_arf_corpus_is_reproducible():
    """Same seed -> same recommended prior emits the same first episode."""
    p1 = recommend_prior(architecture="std", task="regression", seed=7)
    p2 = recommend_prior(architecture="std", task="regression", seed=7)
    X1, y1, _, _ = p1.sample_episode(8, 4, d=4, seed=10)
    X2, y2, _, _ = p2.sample_episode(8, 4, d=4, seed=10)
    assert torch.allclose(X1, X2)


def test_known_tasks_set_contents():
    assert KNOWN_TASKS == {"regression", "classification", "directional"}


def test_known_archs_partition():
    from tabkernels.priors.design import ASYMMETRIC_ARCHS, SYMMETRIC_ARCHS
    assert SYMMETRIC_ARCHS.isdisjoint(ASYMMETRIC_ARCHS)
    assert SYMMETRIC_ARCHS | ASYMMETRIC_ARCHS == KNOWN_ARCHS
