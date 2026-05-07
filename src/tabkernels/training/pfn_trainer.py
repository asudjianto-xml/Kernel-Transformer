"""Prior-Data Fitted Network trainer (Chapter 19).

PFN training: sample episodes from a Prior, train a model to predict
held-out queries given a context. The trained model approximates the
Bayesian posterior predictive p(y_q | x_q, context) under the prior.

References
----------
Müller et al. 2021 — "Transformers Can Do Bayesian Inference".
Hollmann et al. 2022/2023 — TabPFN.
Nagler 2023 — Statistical Foundations of Prior-Fitted Networks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabkernels.core.base import Prior


@dataclass
class PFNTrainState:
    """Snapshot returned by :class:`PFNTrainer`."""
    losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    final_step: int = 0


def regression_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL with unit variance, equivalent to MSE up to a constant.

    For PFN regression the model emits a posterior mean; minimising squared
    error matches the framework's L2 ICL objective.
    """
    return F.mse_loss(y_pred.squeeze(-1), y_true.squeeze(-1))


def classification_loss(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on integer class labels."""
    return F.cross_entropy(logits, y_true.long())


class PFNTrainer:
    """Prior-Data Fitted Network trainer.

    Parameters
    ----------
    prior : Prior
        Episode generator. Each call returns (X_ctx, y_ctx, X_q, y_q).
    model : nn.Module
        Predictor with signature ``model(X_q, X_ctx, y_ctx) -> y_pred``.
    n_steps : int
        Total optimisation steps.
    n_ctx : int
        Context size sampled per episode.
    n_query : int
        Number of held-out queries per episode.
    d : int
        Feature dimension.
    lr : float
        Adam learning rate.
    loss_fn : callable
        ``(y_pred, y_true) -> scalar``. Default: regression MSE.
    eval_every : int
        Run a held-out evaluation every ``eval_every`` steps.
    device : str
        Torch device.
    seed : int | None
        Per-episode seed; if set, episode k uses seed = seed + k.
    """

    def __init__(
        self,
        prior: Prior,
        model: nn.Module,
        n_steps: int = 200,
        n_ctx: int = 64,
        n_query: int = 32,
        d: int = 4,
        lr: float = 1e-3,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = regression_loss,
        eval_every: int = 0,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.prior = prior
        self.model = model.to(device)
        self.n_steps = n_steps
        self.n_ctx = n_ctx
        self.n_query = n_query
        self.d = d
        self.lr = lr
        self.loss_fn = loss_fn
        self.eval_every = eval_every
        self.device = device
        self.seed = seed
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _sample(self, step: int) -> tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor]:
        s = None if self.seed is None else self.seed + step
        X_ctx, y_ctx, X_q, y_q = self.prior.sample_episode(
            n_ctx=self.n_ctx, n_query=self.n_query, d=self.d, seed=s
        )
        return (X_ctx.to(self.device), y_ctx.to(self.device),
                X_q.to(self.device), y_q.to(self.device))

    def step(self, step_idx: int) -> float:
        self.model.train()
        X_ctx, y_ctx, X_q, y_q = self._sample(step_idx)
        y_pred = self.model(X_q, X_ctx, y_ctx)
        loss = self.loss_fn(y_pred, y_q)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 16, eval_seed: int = 9999) -> float:
        self.model.eval()
        total = 0.0
        for k in range(n_episodes):
            X_ctx, y_ctx, X_q, y_q = self.prior.sample_episode(
                n_ctx=self.n_ctx, n_query=self.n_query, d=self.d, seed=eval_seed + k
            )
            X_ctx, y_ctx = X_ctx.to(self.device), y_ctx.to(self.device)
            X_q, y_q = X_q.to(self.device), y_q.to(self.device)
            y_pred = self.model(X_q, X_ctx, y_ctx)
            total += self.loss_fn(y_pred, y_q).item()
        return total / n_episodes

    def train(self) -> PFNTrainState:
        state = PFNTrainState()
        for step in range(self.n_steps):
            loss = self.step(step)
            state.losses.append(loss)
            if self.eval_every and (step + 1) % self.eval_every == 0:
                state.eval_losses.append(self.evaluate())
        state.final_step = self.n_steps
        return state
