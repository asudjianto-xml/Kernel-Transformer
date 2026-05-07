"""Adversarial-Random-Forest prior (Chapter 20 §20.3).

ARF (Watson et al. 2023) fits a generative model by training a random forest
to discriminate real samples from a noise distribution; trees from the forest
then act as a piecewise-uniform sampler over leaves.

Real ARF training requires a labelled corpus and is out of scope here. This
module exposes a *sketch* implementation:

  - Given an in-memory ``corpus`` tensor of shape (n_corpus, d), fit a one-shot
    leaf partition by recursive median splits on random features.
  - Sample new points by picking a leaf uniformly and adding leaf-conditional
    Gaussian noise sized to the leaf's per-feature std.

This captures the *behaviour* the chapter discusses (data-driven feature
distribution, no parametric assumption) without requiring the full ARF
training pipeline. For real ARF training use the ``arf`` CRAN package.
"""
from __future__ import annotations

import torch

from tabkernels.core.base import Prior


def _build_leaf_partition(X: torch.Tensor, max_leaves: int = 32,
                          g: torch.Generator | None = None
                          ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Recursively split X by median on a random feature until we have leaves.

    Returns a list of (leaf_mean, leaf_std) per leaf; leaf size matters less
    than coverage, so we stop when we hit ``max_leaves`` or single-row leaves.
    """
    leaves: list[tuple[torch.Tensor, torch.Tensor]] = []
    work = [X]
    while work and len(leaves) + len(work) < max_leaves:
        chunk = work.pop()
        if chunk.shape[0] <= 4:
            leaves.append((chunk.mean(0), chunk.std(0).clamp_min(1e-3)))
            continue
        d = chunk.shape[1]
        feat = int(torch.randint(d, (1,), generator=g).item()) if g is not None else 0
        med = chunk[:, feat].median()
        left = chunk[chunk[:, feat] <= med]
        right = chunk[chunk[:, feat] > med]
        if left.shape[0] == 0 or right.shape[0] == 0:
            leaves.append((chunk.mean(0), chunk.std(0).clamp_min(1e-3)))
            continue
        work.append(left); work.append(right)
    for chunk in work:
        leaves.append((chunk.mean(0), chunk.std(0).clamp_min(1e-3)))
    return leaves


class ARFPrior(Prior):
    """Adversarial-Random-Forest prior (sketch implementation).

    Parameters
    ----------
    corpus : torch.Tensor
        Seed feature corpus, shape (n_corpus, d). The d here must match the
        d passed to ``sample_episode``.
    label_fn : callable | None
        ``(X) -> y`` mapping features to labels. If None, labels are a fixed
        random linear projection of features (set per-task by seed).
    max_leaves : int
        Cap on the leaf partition.
    noise_scale : float
        Multiplier on leaf-conditional Gaussian noise.
    """

    def __init__(self, corpus: torch.Tensor, label_fn=None,
                 max_leaves: int = 32, noise_scale: float = 1.0):
        if corpus.dim() != 2:
            raise ValueError("corpus must be 2D (n_corpus, d)")
        self.corpus = corpus
        self.label_fn = label_fn
        self.max_leaves = max_leaves
        self.noise_scale = noise_scale

    def sample_episode(self, n_ctx, n_query, d, seed=None):
        if d != self.corpus.shape[1]:
            raise ValueError(f"corpus has d={self.corpus.shape[1]}, requested d={d}")
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        leaves = _build_leaf_partition(self.corpus, max_leaves=self.max_leaves, g=g)
        N = n_ctx + n_query
        leaf_idx = torch.randint(len(leaves), (N,), generator=g)
        out = torch.zeros(N, d)
        for i in range(N):
            mu, sd = leaves[int(leaf_idx[i].item())]
            out[i] = mu + self.noise_scale * sd * torch.randn(d, generator=g)
        if self.label_fn is None:
            w = torch.randn(d, generator=g)
            y = out @ w + 0.1 * torch.randn(N, generator=g)
        else:
            y = self.label_fn(out)
        return out[:n_ctx], y[:n_ctx], out[n_ctx:], y[n_ctx:]
