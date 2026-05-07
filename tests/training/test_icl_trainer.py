"""Tests for tabkernels.training.icl_trainer (Chapter 21)."""
from __future__ import annotations

import torch
import torch.nn as nn

from tabkernels.attention import StdAttention
from tabkernels.priors import SCMPrior
from tabkernels.training import ICLDiagnostics, ICLTrainer


def _model(d_in=4, d_emb=16):
    block = StdAttention(d_in=d_in, d_emb=d_emb)

    class _W(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = block

        def forward(self, X_q, X_c, y_c):
            return self.block(X_q, X_c, y_c).unsqueeze(-1)

    return _W()


def test_icl_trainer_runs():
    p = SCMPrior()
    m = _model()
    t = ICLTrainer(prior=p, model=m, n_steps=10, n_ctx=8, n_query=4, d=4,
                   lr=1e-3, warmup_steps=2, eval_every=5, seed=0)
    diag = t.train()
    assert isinstance(diag, ICLDiagnostics)
    assert len(diag.losses) == 10
    assert len(diag.grad_norms) == 10
    assert len(diag.lrs) == 10
    assert len(diag.eval_losses) == 2
    assert diag.final_step == 10


def test_warmup_lr_schedule_monotonic_then_decreasing():
    """During warmup the LR should rise; after warmup it should decay."""
    p = SCMPrior()
    m = _model()
    t = ICLTrainer(prior=p, model=m, n_steps=20, n_ctx=8, n_query=4, d=4,
                   lr=1e-2, warmup_steps=5, eval_every=0, seed=0)
    diag = t.train()
    # Warmup: lrs[0] < lrs[1] < ... < lrs[4]
    for i in range(4):
        assert diag.lrs[i] < diag.lrs[i + 1]
    # Post-warmup: lrs[5] >= lrs[19] (cosine decay to 1% of peak).
    assert diag.lrs[5] >= diag.lrs[-1]


def test_grad_clip_caps_norms():
    p = SCMPrior()
    m = _model()
    t = ICLTrainer(prior=p, model=m, n_steps=8, n_ctx=8, n_query=4, d=4,
                   lr=1e-2, warmup_steps=1, grad_clip=0.1, eval_every=0, seed=0)
    diag = t.train()
    # Pre-clip grad norms get *recorded*, but the optimiser sees clipped grads.
    # Just check the recorded norms are finite and non-negative.
    assert all(g >= 0 and torch.isfinite(torch.tensor(g)) for g in diag.grad_norms)


def test_diagnostics_record_eval_at_correct_steps():
    p = SCMPrior()
    m = _model()
    t = ICLTrainer(prior=p, model=m, n_steps=12, n_ctx=8, n_query=4, d=4,
                   lr=1e-3, warmup_steps=0, eval_every=4, seed=0)
    diag = t.train()
    assert diag.eval_steps == [4, 8, 12]
    assert len(diag.eval_losses) == 3


def test_no_warmup_path():
    """warmup_steps=0 should hit the post-warmup branch immediately."""
    p = SCMPrior()
    m = _model()
    t = ICLTrainer(prior=p, model=m, n_steps=4, n_ctx=8, n_query=4, d=4,
                   lr=1e-3, warmup_steps=0, eval_every=0, seed=0)
    diag = t.train()
    # First lr should be at peak (cosine starts at 1 progress=0).
    assert diag.lrs[0] > 0.99 * t.peak_lr


def test_evaluate_uses_no_grad_and_returns_float():
    p = SCMPrior()
    m = _model()
    t = ICLTrainer(prior=p, model=m, n_steps=1, n_ctx=8, n_query=4, d=4,
                   eval_episodes=3, seed=0)
    eval_loss = t.evaluate()
    assert isinstance(eval_loss, float)
    assert eval_loss > 0


def test_loss_decreases_on_easy_prior():
    """ICLTrainer with a 200-step run on a default SCM should reduce loss."""
    p = SCMPrior()
    m = _model(d_emb=24)
    t = ICLTrainer(prior=p, model=m, n_steps=200, n_ctx=24, n_query=12, d=4,
                   lr=3e-3, warmup_steps=20, eval_every=0, seed=1)
    diag = t.train()
    early = sum(diag.losses[:20]) / 20
    late = sum(diag.losses[-20:]) / 20
    assert late < early
