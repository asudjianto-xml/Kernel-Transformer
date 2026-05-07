"""Synthetic priors for tabular ICL (Chapter 20).

SCMPrior              — structural-causal-model prior (TabPFN-style).
ARFPrior              — Adversarial-Random-Forest sketch (Watson et al. 2023).
MLPSCMPrior           — feature distribution + random-MLP labels (our hybrid).
KnowledgeGraphPrior   — KG-augmented prior sketch (TARTE/TabSTAR-style).
"""
from tabkernels.priors.scm import SCMPrior, SCMConfig
from tabkernels.priors.arf import ARFPrior
from tabkernels.priors.mlp_scm import MLPSCMPrior
from tabkernels.priors.knowledge_graph import KnowledgeGraphPrior, ConceptHierarchy
from tabkernels.priors.design import recommend_prior, KNOWN_ARCHS, KNOWN_TASKS

__all__ = [
    "SCMPrior",
    "SCMConfig",
    "ARFPrior",
    "MLPSCMPrior",
    "KnowledgeGraphPrior",
    "ConceptHierarchy",
    "recommend_prior",
    "KNOWN_ARCHS",
    "KNOWN_TASKS",
]
