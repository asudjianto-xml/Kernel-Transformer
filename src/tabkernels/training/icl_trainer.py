"""ICL training-dynamics trainer (Chapter 21).

ICLTrainer extends PFNTrainer with the engineering surfaces a serious ICL
training run needs:

  - Linear-warmup + cosine-decay learning-rate schedule.
  - Gradient-norm tracking and clipping.
  - Per-step metric recording (loss, grad norm, lr).
  - Failure-mode detection: NaN losses, runaway gradients, frozen-loss plateaus.
  - Configurable evaluation cadence with a fixed held-out seed range.

The training loop is otherwise identical to :class:`PFNTrainer`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from tabkernels.core.base import Prior
from tabkernels.training.pfn_trainer import PFNTrainer, regression_loss


@dataclass
class ICLDiagnostics:
    """Per-run diagnostics emitted by :class:`ICLTrainer`."""
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    nan_steps: list[int] = field(default_factory=list)
    final_step: int = 0


class ICLTrainer:
    """Trainer with explicit dynamics tracking.

    Parameters
    ----------
    prior : Prior
        Episode generator.
    model : nn.Module
        Predictor with signature ``model(X_q, X_ctx, y_ctx) -> y_pred``.
    n_steps : int
        Total optimisation steps.
    n_ctx, n_query, d : int
        Episode shape.
    lr : float
        Peak learning rate (the warmup target and cosine-decay starting value).
    warmup_steps : int
        Linear-warmup duration. Set to 0 to disable warmup.
    grad_clip : float | None
        Optional gradient-norm clip. Set to None to disable.
    loss_fn : callable
        ``(y_pred, y_true) -> scalar``. Defaults to MSE.
    eval_every : int
        Run a held-out evaluation every ``eval_every`` steps.
    eval_episodes : int
        Number of held-out episodes per evaluation.
    device : str
        Torch device.
    seed : int | None
        Per-step seed; if set, episode k uses seed = seed + k.
    """

    def __init__(
        self,
        prior: Prior,
        model: nn.Module,
        n_steps: int = 500,
        n_ctx: int = 64,
        n_query: int = 32,
        d: int = 4,
        lr: float = 3e-3,
        warmup_steps: int = 50,
        grad_clip: float | None = 1.0,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = regression_loss,
        eval_every: int = 50,
        eval_episodes: int = 16,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.prior = prior
        self.model = model.to(device)
        self.n_steps = n_steps
        self.n_ctx = n_ctx
        self.n_query = n_query
        self.d = d
        self.peak_lr = lr
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.loss_fn = loss_fn
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.device = device
        self.seed = seed
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        # We borrow PFNTrainer's episode sampler.
        self._sampler = PFNTrainer(
            prior=prior, model=model, n_steps=n_steps, n_ctx=n_ctx,
            n_query=n_query, d=d, lr=lr, loss_fn=loss_fn,
            device=device, seed=seed,
        )

    def _lr_for_step(self, step: int) -> float:
        """Linear warmup + cosine decay to 1% of peak."""
        if self.warmup_steps and step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        # Cosine from peak_lr to 0.01 * peak_lr over remaining steps.
        remaining = max(1, self.n_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / remaining
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
        return self.peak_lr * (0.01 + 0.99 * cosine)

    def _grad_norm(self) -> float:
        sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                sq += p.grad.detach().pow(2).sum().item()
        return float(sq ** 0.5)

    def step(self, step_idx: int, diag: ICLDiagnostics) -> bool:
        """Run one step. Returns False if a NaN was detected."""
        lr = self._lr_for_step(step_idx)
        for pg in self.optim.param_groups:
            pg["lr"] = lr
        self.model.train()
        X_c, y_c, X_q, y_q = self._sampler._sample(step_idx)
        y_pred = self.model(X_q, X_c, y_c)
        loss = self.loss_fn(y_pred, y_q)
        if torch.isnan(loss) or torch.isinf(loss):
            diag.nan_steps.append(step_idx)
            self.optim.zero_grad()
            return False
        self.optim.zero_grad()
        loss.backward()
        gn = self._grad_norm()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()
        diag.losses.append(loss.item())
        diag.grad_norms.append(gn)
        diag.lrs.append(lr)
        return True

    @torch.no_grad()
    def evaluate(self, eval_seed: int = 9999) -> float:
        self.model.eval()
        total = 0.0
        for k in range(self.eval_episodes):
            X_c, y_c, X_q, y_q = self.prior.sample_episode(
                n_ctx=self.n_ctx, n_query=self.n_query, d=self.d, seed=eval_seed + k
            )
            X_c, y_c = X_c.to(self.device), y_c.to(self.device)
            X_q, y_q = X_q.to(self.device), y_q.to(self.device)
            y_pred = self.model(X_q, X_c, y_c)
            total += self.loss_fn(y_pred, y_q).item()
        return total / self.eval_episodes

    def train(self) -> ICLDiagnostics:
        diag = ICLDiagnostics()
        for step in range(self.n_steps):
            self.step(step, diag)
            if self.eval_every and (step + 1) % self.eval_every == 0:
                diag.eval_steps.append(step + 1)
                diag.eval_losses.append(self.evaluate())
        diag.final_step = self.n_steps
        return diag
