"""Base classes that lock the package's API contract.

Every kernel, sparsifier, symmetrizer, predictor, attention, and audit in the
package inherits from one of these. Chapter agents subclass these; they do not
modify them.
"""
from __future__ import annotations

import abc

import torch
import torch.nn as nn

from .types import FeatureTensor, GramTensor, TargetTensor


class Kernel(nn.Module, abc.ABC):
    """A bivariate kernel function.

    Subclasses must implement `forward(X1, X2) -> GramTensor` returning the
    (N1, N2) Gram matrix k(X1, X2).
    """

    @abc.abstractmethod
    def forward(self, X1: FeatureTensor, X2: FeatureTensor) -> GramTensor:
        """Return the (N1, N2) kernel matrix between rows of X1 and X2."""
        ...

    def is_symmetric(self) -> bool:
        """Whether the kernel is symmetric: k(x, y) = k(y, x). Default: True."""
        return True

    def is_psd(self) -> bool:
        """Whether the kernel is positive semi-definite. Default: True."""
        return True


class Sparsifier(nn.Module, abc.ABC):
    """An operator that takes a Gram matrix and produces a sparse one.

    Subclasses must implement `forward(W) -> sparsified W`.
    """

    @abc.abstractmethod
    def forward(self, W: GramTensor) -> GramTensor:
        """Return a sparsified version of W with the same shape."""
        ...

    def is_differentiable(self) -> bool:
        """Whether gradients flow through the sparsification. Default: True."""
        return True


class Symmetrizer(nn.Module, abc.ABC):
    """An operator that symmetrizes an asymmetric Gram matrix.

    Subclasses must implement `forward(W) -> symmetric W`.
    """

    @abc.abstractmethod
    def forward(self, W: GramTensor) -> GramTensor:
        """Return a symmetric (N, N) matrix derived from W."""
        ...


class Predictor(nn.Module, abc.ABC):
    """Takes weights W and training labels y_train; returns predictions.

    Subclasses must implement `forward(W, y_train) -> predictions`.
    """

    @abc.abstractmethod
    def forward(self, W: GramTensor, y_train: TargetTensor) -> TargetTensor:
        """Return predictions for the queries indexed by rows of W."""
        ...


class AttentionBlock(nn.Module, abc.ABC):
    """A complete kernel-attention block: Kernel + (optional Sparsifier) +
    (optional Symmetrizer) + Predictor.

    All five attention variants in `tabkernels.attention` (Std, SymPSD, SymGen,
    PureAsym, Dual) inherit from this. Chapter 13 (transparency) requires
    every AttentionBlock to expose `decompose()` and `energy_split()`.
    """

    @abc.abstractmethod
    def forward(self, X_q: FeatureTensor, X_t: FeatureTensor,
                y_t: TargetTensor) -> TargetTensor:
        """Predict y at X_q using training (X_t, y_t)."""
        ...

    @abc.abstractmethod
    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B_S, B_A): the symmetric and skew-symmetric components of the
        learned bilinear form B in input space.

        For models without a single learned B (e.g., dual-channel), return the
        sum of the components from each branch.
        """
        ...

    def energy_split(self) -> tuple[float, float]:
        """Return (alpha_S, alpha_A): Frobenius energy fractions of B_S and B_A.

        Default implementation calls self.decompose(). Subclasses may override
        for efficiency.
        """
        B_S, B_A = self.decompose()
        s = (B_S ** 2).sum().item()
        a = (B_A ** 2).sum().item()
        total = s + a + 1e-12
        return s / total, a / total


class Architecture(nn.Module, abc.ABC):
    """A complete tabular-prediction architecture (FT-Transformer, SAINT,
    TabPFN-lite, TabICL-lite, etc.).

    Defines the contract for full architectures: forward, train_step, predict.
    """

    @abc.abstractmethod
    def forward(self, X: FeatureTensor) -> torch.Tensor: ...

    def attention_blocks(self) -> list[AttentionBlock]:
        """Return all attention blocks in the architecture, for use by the
        Ch 13 transparency tooling. Default: empty list."""
        return []


class Prior(abc.ABC):
    """A synthetic-data prior for ICL pretraining.

    Subclasses (SCM, ARF, MLP-SCM, knowledge-graph) implement `sample_episode`.
    """

    @abc.abstractmethod
    def sample_episode(self, n_ctx: int, n_query: int, d: int,
                       seed: int | None = None
                       ) -> tuple[FeatureTensor, TargetTensor,
                                  FeatureTensor, TargetTensor]:
        """Return (X_ctx, y_ctx, X_query, y_query)."""
        ...


class Audit(abc.ABC):
    """A diagnostic that takes a trained model + data and produces a report.

    Subclasses (SymmetryAudit, OracleAudit, PairwiseAudit) implement `run`.
    """

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> dict:
        """Return a dict-shaped report of audit results."""
        ...
