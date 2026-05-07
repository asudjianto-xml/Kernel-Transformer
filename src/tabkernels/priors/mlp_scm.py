"""MLP-SCM hybrid prior (Chapter 20 §20.4).

Our hybrid prior:

  - Features X come from any feature distribution (e.g., ARF for realism).
  - Labels y come from a small randomly-initialised MLP applied to X.

The motivation is to decouple two engineering levers:

  - Realistic feature marginals require something data-driven (ARF).
  - Controllable label structure requires something parametric (MLP-SCM).

Pure SCM priors give controllability on both fronts but at the cost of
unrealistic feature marginals. ARF gives realistic features but the
label-generation mechanism is constrained to whatever ARF infers from the
corpus. The hybrid takes one component from each.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tabkernels.core.base import Prior


class _RandomMLP(nn.Module):
    """A randomly-initialised MLP used as a label-generating SCM."""

    def __init__(self, d_in: int, hidden: int, depth: int,
                 activation: str = "tanh", g: torch.Generator | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d_in
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act_cls())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        # Re-seed init from g.
        if g is not None:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    fan_in = m.weight.shape[1]
                    scale = (2.0 / fan_in) ** 0.5
                    m.weight.data = scale * torch.randn(m.weight.shape, generator=g)
                    m.bias.data = 0.1 * torch.randn(m.bias.shape, generator=g)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)


class MLPSCMPrior(Prior):
    """Feature distribution + random-MLP label generator.

    Parameters
    ----------
    feature_prior : Prior
        Any prior whose features will be used as inputs to the MLP-SCM.
        Its labels are discarded.
    hidden : int
        MLP hidden width.
    depth : int
        Number of hidden layers.
    activation : str
        ``"tanh"``, ``"relu"``, or ``"gelu"``.
    noise_scale : float
        Std of additive Gaussian noise on labels.
    """

    def __init__(self, feature_prior: Prior, hidden: int = 16, depth: int = 2,
                 activation: str = "tanh", noise_scale: float = 0.1):
        self.feature_prior = feature_prior
        self.hidden = hidden
        self.depth = depth
        self.activation = activation
        self.noise_scale = noise_scale

    def sample_episode(self, n_ctx, n_query, d, seed=None):
        # Get features from the underlying feature prior; ignore its labels.
        X_c, _, X_q, _ = self.feature_prior.sample_episode(
            n_ctx=n_ctx, n_query=n_query, d=d, seed=seed
        )
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed + 1234567)
        mlp = _RandomMLP(d_in=d, hidden=self.hidden, depth=self.depth,
                         activation=self.activation, g=g)
        with torch.no_grad():
            X_full = torch.cat([X_c, X_q], dim=0)
            y_full = mlp(X_full)
            y_full = y_full + self.noise_scale * torch.randn(y_full.shape[0], generator=g)
        return X_c, y_full[:n_ctx], X_q, y_full[n_ctx:]
