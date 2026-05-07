"""Tests for tabkernels.training.pfn_trainer (Chapter 19)."""
from __future__ import annotations

import torch
import torch.nn as nn

from tabkernels.attention import StdAttention
from tabkernels.core.base import Prior
from tabkernels.training import (
    PFNTrainer,
    PFNTrainState,
    classification_loss,
    regression_loss,
)


class _LinearRegPrior(Prior):
    """Bayesian linear-regression prior with closed-form posterior.

    Each episode samples a task weight w ~ N(0, sigma_w^2 I_d) and emits
    (X, y = X w + eps) with eps ~ N(0, sigma_eps^2).
    """

    def __init__(self, sigma_w: float = 1.0, sigma_eps: float = 0.1):
        self.sigma_w = sigma_w
        self.sigma_eps = sigma_eps

    def sample_episode(self, n_ctx, n_query, d, seed=None):
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        w = self.sigma_w * torch.randn(d, generator=g)
        N = n_ctx + n_query
        X = torch.randn(N, d, generator=g)
        eps = self.sigma_eps * torch.randn(N, generator=g)
        y = X @ w + eps
        return X[:n_ctx], y[:n_ctx], X[n_ctx:], y[n_ctx:]


def _attention_predictor(d_in: int = 4, d_emb: int = 8) -> nn.Module:
    """Wrap StdAttention to return a (n_query, 1)-shaped tensor."""
    block = StdAttention(d_in=d_in, d_emb=d_emb)

    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = block

        def forward(self, X_q, X_ctx, y_ctx):
            return self.block(X_q, X_ctx, y_ctx).unsqueeze(-1)

    return _Wrap()


def test_trainer_constructs():
    prior = _LinearRegPrior()
    model = _attention_predictor()
    trainer = PFNTrainer(prior=prior, model=model, n_steps=2, n_ctx=8,
                         n_query=4, d=4)
    assert trainer.n_steps == 2


def test_single_step_returns_loss():
    prior = _LinearRegPrior()
    model = _attention_predictor()
    trainer = PFNTrainer(prior=prior, model=model, n_steps=1, n_ctx=8,
                         n_query=4, d=4, seed=0)
    loss = trainer.step(0)
    assert isinstance(loss, float)
    assert loss > 0


def test_train_reduces_loss():
    """Loss should drop materially over a short training run."""
    torch.manual_seed(0)
    prior = _LinearRegPrior(sigma_w=1.0, sigma_eps=0.1)
    model = _attention_predictor(d_in=4, d_emb=16)
    trainer = PFNTrainer(prior=prior, model=model, n_steps=80, n_ctx=32,
                         n_query=16, d=4, lr=5e-3, seed=1)
    state = trainer.train()
    early = sum(state.losses[:10]) / 10
    late = sum(state.losses[-10:]) / 10
    assert late < early


def test_evaluate_runs_no_grad():
    prior = _LinearRegPrior()
    model = _attention_predictor()
    trainer = PFNTrainer(prior=prior, model=model, n_steps=1, n_ctx=8,
                         n_query=4, d=4)
    eval_loss = trainer.evaluate(n_episodes=3)
    assert isinstance(eval_loss, float)
    assert eval_loss > 0


def test_eval_every_records_eval_losses():
    prior = _LinearRegPrior()
    model = _attention_predictor()
    trainer = PFNTrainer(prior=prior, model=model, n_steps=4, n_ctx=8,
                         n_query=4, d=4, eval_every=2, seed=0)
    state = trainer.train()
    assert len(state.losses) == 4
    assert len(state.eval_losses) == 2


def test_seeded_reproducibility():
    """Two trainers with the same seed produce the same loss sequence."""
    p1 = _LinearRegPrior(); p2 = _LinearRegPrior()
    torch.manual_seed(0); m1 = _attention_predictor()
    torch.manual_seed(0); m2 = _attention_predictor()
    t1 = PFNTrainer(prior=p1, model=m1, n_steps=5, n_ctx=8,
                    n_query=4, d=4, lr=1e-3, seed=7)
    t2 = PFNTrainer(prior=p2, model=m2, n_steps=5, n_ctx=8,
                    n_query=4, d=4, lr=1e-3, seed=7)
    s1 = t1.train(); s2 = t2.train()
    for a, b in zip(s1.losses, s2.losses):
        assert abs(a - b) < 1e-5


def test_state_records_steps():
    prior = _LinearRegPrior()
    model = _attention_predictor()
    trainer = PFNTrainer(prior=prior, model=model, n_steps=3, n_ctx=8,
                         n_query=4, d=4, seed=0)
    state = trainer.train()
    assert isinstance(state, PFNTrainState)
    assert state.final_step == 3
    assert len(state.losses) == 3


def test_classification_loss_signature():
    logits = torch.randn(4, 3)
    y = torch.tensor([0, 1, 2, 1])
    loss = classification_loss(logits, y)
    assert loss.dim() == 0


def test_regression_loss_squeezes():
    y_pred = torch.randn(5, 1)
    y_true = torch.randn(5)
    loss = regression_loss(y_pred, y_true)
    assert loss.dim() == 0
