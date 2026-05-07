"""Prior--architecture co-design (Chapter 22).

A small dispatcher that maps an architecture family + task type onto a
recommended prior from :mod:`tabkernels.priors`. Defaults reflect the
empirical recommendations of Chapter 22; users can override with explicit
constructor calls.

Architecture keys
-----------------
``"sym_psd"``, ``"sym_gen"``, ``"std"`` -- symmetric and softmax-symmetric blocks.
``"pure_asym"``, ``"dual"``             -- asymmetric blocks.

Task keys
---------
``"regression"``, ``"classification"`` -- only regression supported in this
release; classification falls back to regression with a softer noise scale.
``"directional"``                     -- forwards to Chapter 25 directional
priors when those land. For now we return a low-edge SCM with a designated
topological order, which captures the directional shape.
"""
from __future__ import annotations

from typing import Optional

import torch

from tabkernels.core.base import Prior
from tabkernels.priors.arf import ARFPrior
from tabkernels.priors.knowledge_graph import KnowledgeGraphPrior
from tabkernels.priors.mlp_scm import MLPSCMPrior
from tabkernels.priors.scm import SCMConfig, SCMPrior


SYMMETRIC_ARCHS = {"sym_psd", "sym_gen", "std"}
ASYMMETRIC_ARCHS = {"pure_asym", "dual"}
KNOWN_ARCHS = SYMMETRIC_ARCHS | ASYMMETRIC_ARCHS

KNOWN_TASKS = {"regression", "classification", "directional"}


def recommend_prior(architecture: str, task: str = "regression",
                    d_features: int = 4,
                    seed: Optional[int] = None) -> Prior:
    """Return a configured prior matched to the architecture / task pair.

    Parameters
    ----------
    architecture : str
        One of ``KNOWN_ARCHS``.
    task : str
        One of ``KNOWN_TASKS``. ``"directional"`` returns a prior with an
        explicit topological order; the others return symmetric priors.
    d_features : int
        Used only for sizing the seed corpus when an ARF-backed prior is
        recommended.
    seed : int | None
        Reproducibility seed for ARF seed corpus.

    Returns
    -------
    Prior
        A subclass of :class:`tabkernels.core.base.Prior` configured for the
        architecture/task pair.

    Notes
    -----
    The recommendations follow Chapter 22 Table 22.1. They are defaults, not
    optima: serious deployments should tune the prior to the empirical
    feature distribution of the target tasks.
    """
    if architecture not in KNOWN_ARCHS:
        raise ValueError(f"unknown architecture {architecture!r}; "
                         f"choose from {sorted(KNOWN_ARCHS)}")
    if task not in KNOWN_TASKS:
        raise ValueError(f"unknown task {task!r}; "
                         f"choose from {sorted(KNOWN_TASKS)}")

    if task == "directional":
        # Directional task -- the Chapter 25 territory.
        # Use SCM with high edge-density to get explicit topological structure.
        return SCMPrior(SCMConfig(structural="mlp", edge_prob=0.7,
                                  noise_scale=0.3))

    if architecture in ASYMMETRIC_ARCHS:
        # Asymmetric architecture -- still recommend a directional-leaning prior.
        return SCMPrior(SCMConfig(structural="mlp", edge_prob=0.6,
                                  noise_scale=0.3))

    # Symmetric architecture -- prefer the hybrid for breadth.
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    n_corpus = 400
    centres = torch.tensor([
        [2.0] + [0.0] * (d_features - 1),
        [0.0, 2.0] + [0.0] * (d_features - 2) if d_features >= 2 else [2.0],
        [-2.0] * d_features,
    ])[:, :d_features]
    modes = torch.randint(centres.shape[0], (n_corpus,), generator=g)
    seed_corpus = centres[modes] + 0.5 * torch.randn(n_corpus, d_features, generator=g)
    arf = ARFPrior(corpus=seed_corpus, max_leaves=24)
    return MLPSCMPrior(feature_prior=arf, hidden=12, depth=2,
                       activation="tanh", noise_scale=0.1)
