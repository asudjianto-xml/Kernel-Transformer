"""Tests for tabkernels.priors (Chapter 20)."""
from __future__ import annotations

import torch

from tabkernels.priors import (
    ARFPrior,
    KnowledgeGraphPrior,
    MLPSCMPrior,
    SCMConfig,
    SCMPrior,
)


def _check_episode_shapes(prior, n_ctx=8, n_query=4, d=3, seed=0):
    X_c, y_c, X_q, y_q = prior.sample_episode(n_ctx=n_ctx, n_query=n_query, d=d, seed=seed)
    assert X_c.shape == (n_ctx, d)
    assert y_c.shape == (n_ctx,)
    assert X_q.shape == (n_query, d)
    assert y_q.shape == (n_query,)
    assert torch.isfinite(X_c).all() and torch.isfinite(y_c).all()
    assert torch.isfinite(X_q).all() and torch.isfinite(y_q).all()


def test_scm_default_shapes():
    _check_episode_shapes(SCMPrior())


def test_scm_mlp_structural():
    _check_episode_shapes(SCMPrior(SCMConfig(structural="mlp", mlp_hidden=8)))


def test_scm_seed_reproducible():
    p = SCMPrior()
    a = p.sample_episode(8, 4, 3, seed=42)
    b = p.sample_episode(8, 4, 3, seed=42)
    for x, y in zip(a, b):
        assert torch.allclose(x, y)


def test_scm_dropout_zeros_columns():
    """With dropout=1.0, every feature column should be zero."""
    p = SCMPrior(SCMConfig(feature_dropout=1.0))
    X_c, _, _, _ = p.sample_episode(16, 4, 4, seed=0)
    assert (X_c == 0).all()


def test_arf_with_corpus():
    torch.manual_seed(0)
    corpus = torch.randn(80, 3)
    p = ARFPrior(corpus=corpus)
    _check_episode_shapes(p, d=3)


def test_arf_dimension_mismatch_raises():
    p = ARFPrior(corpus=torch.randn(50, 3))
    raised = False
    try:
        p.sample_episode(4, 2, d=4)
    except ValueError:
        raised = True
    assert raised


def test_arf_2d_corpus_required():
    raised = False
    try:
        ARFPrior(corpus=torch.randn(50))
    except ValueError:
        raised = True
    assert raised


def test_mlp_scm_inherits_features_from_inner_prior():
    """MLPSCM should reuse the inner prior's features verbatim."""
    inner = SCMPrior()
    hybrid = MLPSCMPrior(feature_prior=inner, hidden=8, depth=2)
    X_c1, _, _, _ = inner.sample_episode(8, 4, 3, seed=99)
    X_c2, _, _, _ = hybrid.sample_episode(8, 4, 3, seed=99)
    assert torch.allclose(X_c1, X_c2)


def test_mlp_scm_labels_change_per_seed():
    inner = SCMPrior()
    hybrid = MLPSCMPrior(feature_prior=inner, hidden=8, depth=2)
    _, y_a, _, _ = hybrid.sample_episode(16, 4, 3, seed=1)
    _, y_b, _, _ = hybrid.sample_episode(16, 4, 3, seed=2)
    assert not torch.allclose(y_a, y_b)


def test_knowledge_graph_shapes():
    _check_episode_shapes(KnowledgeGraphPrior(n_concepts=16, d_embed=8))


def test_knowledge_graph_labels_are_integers_plus_noise():
    """Labels are concept depths plus small noise; means should be a positive integer-ish value."""
    p = KnowledgeGraphPrior(n_concepts=32, d_embed=8, noise_scale=0.01)
    _, y_c, _, _ = p.sample_episode(64, 4, 4, seed=0)
    # Depths are non-negative, and noise is tiny, so y should be >= -0.5 or so.
    assert y_c.min() > -0.5


def test_all_priors_subclass_prior_base():
    from tabkernels.core.base import Prior
    assert issubclass(SCMPrior, Prior)
    assert issubclass(ARFPrior, Prior)
    assert issubclass(MLPSCMPrior, Prior)
    assert issubclass(KnowledgeGraphPrior, Prior)
