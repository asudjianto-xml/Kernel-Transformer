"""Structural-causal-model prior (Chapter 20 §20.2).

Each episode:
  1. Sample a random DAG on `d` nodes.
  2. Assign each node a structural equation (linear with noise, or a small MLP).
  3. Generate observations by topological order.
  4. Designate one node as the target y; the rest are features X.

This is the recipe behind TabPFN's prior. We expose the knobs that matter
for downstream ICL: density of edges, structural-equation family, noise
scale, and whether to dropout features at sample time.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from tabkernels.core.base import Prior


@dataclass
class SCMConfig:
    edge_prob: float = 0.5
    noise_scale: float = 0.5
    structural: str = "linear"  # 'linear' | 'mlp'
    mlp_hidden: int = 16
    feature_dropout: float = 0.0


class SCMPrior(Prior):
    """Structural-causal-model prior.

    Parameters
    ----------
    cfg : SCMConfig | None
        Controls edge density, noise scale, structural-equation family,
        and feature dropout. Defaults to ``SCMConfig()``.
    """

    def __init__(self, cfg: SCMConfig | None = None):
        self.cfg = cfg or SCMConfig()

    def _sample_dag(self, d: int, g: torch.Generator) -> torch.Tensor:
        """Random topological-order DAG. Returns adjacency (d, d), upper-triangular."""
        A = torch.zeros(d, d)
        for i in range(d):
            for j in range(i + 1, d):
                if torch.rand((), generator=g).item() < self.cfg.edge_prob:
                    A[i, j] = 1.0
        return A

    def _structural_step(self, parent_vals: torch.Tensor, n: int,
                         g: torch.Generator) -> torch.Tensor:
        """Compute a single node's value given its parents' values (n,) per parent."""
        if parent_vals.numel() == 0:
            return torch.randn(n, generator=g)
        n_parents = parent_vals.shape[1]
        if self.cfg.structural == "linear":
            w = torch.randn(n_parents, generator=g)
            out = parent_vals @ w
        else:  # mlp
            h = self.cfg.mlp_hidden
            W1 = torch.randn(n_parents, h, generator=g) / max(1.0, n_parents ** 0.5)
            b1 = torch.randn(h, generator=g) * 0.1
            W2 = torch.randn(h, generator=g) / (h ** 0.5)
            out = torch.tanh(parent_vals @ W1 + b1) @ W2
        out = out + self.cfg.noise_scale * torch.randn(n, generator=g)
        return out

    def sample_episode(self, n_ctx, n_query, d, seed=None):
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        N = n_ctx + n_query
        # +1 because we'll designate one node as y. So sample d+1 columns.
        d_total = d + 1
        A = self._sample_dag(d_total, g)
        cols = torch.zeros(N, d_total)
        for j in range(d_total):
            parents = torch.where(A[:, j] > 0)[0]
            cols[:, j] = self._structural_step(cols[:, parents], N, g)
        # Pick the most-connected node (most incoming edges) as y so it's predictable.
        in_degrees = A.sum(dim=0)
        y_idx = int(torch.argmax(in_degrees).item())
        if in_degrees[y_idx].item() == 0:
            y_idx = d_total - 1
        feature_idx = [j for j in range(d_total) if j != y_idx]
        X_all = cols[:, feature_idx]
        y_all = cols[:, y_idx]
        if self.cfg.feature_dropout > 0:
            mask = (torch.rand(d, generator=g) > self.cfg.feature_dropout).float()
            X_all = X_all * mask
        return X_all[:n_ctx], y_all[:n_ctx], X_all[n_ctx:], y_all[n_ctx:]
