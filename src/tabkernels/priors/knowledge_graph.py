"""Knowledge-graph-augmented prior (Chapter 20 §20.5, sketch).

Real knowledge-graph priors (TARTE, TabSTAR-style verbalisation) require an
external KG and an embedding model and are out of scope here. This module
provides a *sketch* that captures the shape of the construction:

  - A small synthetic concept hierarchy stands in for the KG.
  - Each "concept" has a learned embedding vector.
  - Features include random projections of the relevant concept embedding.

The point is to demonstrate that a KG-augmented prior is just an SCM whose
exogenous-noise channel is replaced by a structured-vocabulary lookup --- the
architectural primitive is the same, only the source of the inputs changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tabkernels.core.base import Prior


@dataclass
class ConceptHierarchy:
    """A toy concept hierarchy: each concept points to a parent (or -1 = root)."""
    parents: list[int]   # len = n_concepts
    embeddings: torch.Tensor  # (n_concepts, d_embed)

    @classmethod
    def random_tree(cls, n_concepts: int, d_embed: int,
                    g: torch.Generator | None = None) -> "ConceptHierarchy":
        parents = [-1]
        for i in range(1, n_concepts):
            if g is None:
                p = torch.randint(0, i, (1,)).item()
            else:
                p = torch.randint(0, i, (1,), generator=g).item()
            parents.append(int(p))
        if g is None:
            emb = torch.randn(n_concepts, d_embed)
        else:
            emb = torch.randn(n_concepts, d_embed, generator=g)
        # Inherit half of the parent's embedding to put hierarchy structure into similarity.
        for i in range(1, n_concepts):
            emb[i] = 0.5 * emb[parents[i]] + 0.5 * emb[i]
        return cls(parents=parents, embeddings=emb)


class KnowledgeGraphPrior(Prior):
    """KG-augmented prior (sketch).

    Parameters
    ----------
    n_concepts : int
        Size of the synthetic concept vocabulary.
    d_embed : int
        Embedding dimension. Features are a random projection from this space.
    noise_scale : float
        Per-feature additive Gaussian noise.
    """

    def __init__(self, n_concepts: int = 32, d_embed: int = 16,
                 noise_scale: float = 0.1):
        self.n_concepts = n_concepts
        self.d_embed = d_embed
        self.noise_scale = noise_scale

    def sample_episode(self, n_ctx, n_query, d, seed=None):
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        hierarchy = ConceptHierarchy.random_tree(
            n_concepts=self.n_concepts, d_embed=self.d_embed, g=g
        )
        N = n_ctx + n_query
        concept_ids = torch.randint(self.n_concepts, (N,), generator=g)
        emb = hierarchy.embeddings[concept_ids]  # (N, d_embed)
        proj = torch.randn(self.d_embed, d, generator=g) / (self.d_embed ** 0.5)
        X = emb @ proj + self.noise_scale * torch.randn(N, d, generator=g)
        # Label: depth in the hierarchy + noise --- a cheap "structural" signal.
        depth = torch.zeros(N)
        for i in range(N):
            c = int(concept_ids[i].item())
            d_i = 0
            while hierarchy.parents[c] != -1:
                c = hierarchy.parents[c]; d_i += 1
            depth[i] = float(d_i)
        y = depth + self.noise_scale * torch.randn(N, generator=g)
        return X[:n_ctx], y[:n_ctx], X[n_ctx:], y[n_ctx:]
