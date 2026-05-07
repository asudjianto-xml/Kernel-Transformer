"""Element-wise symmetrisation operators (Chapter 6 §6.2-§6.4)."""
from __future__ import annotations

import torch

from tabkernels.core.base import Symmetrizer


class AdditiveSymmetrizer(Symmetrizer):
    """W' = (W + W^T) / 2 (Eq. 6.1).

    Cheap; preserves PSD if input was PSD; tends to densify the support.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square, got {tuple(W.shape)}")
        return 0.5 * (W + W.transpose(-2, -1))


class MaxOrSymmetrizer(Symmetrizer):
    """W' = max(W, W^T) (Eq. 6.2).

    Standard in k-NN graph constructions; biased toward denser graphs.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return torch.maximum(W, W.transpose(-2, -1))


class MutualAndSymmetrizer(Symmetrizer):
    """W' = W * (W^T > 0) * (W > 0) (Eq. 6.3).

    AND-symmetrisation: keeps only mutual edges. Biased toward sparser
    graphs; can disconnect components.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        mask_W = (W > 0).to(W.dtype)
        mask_WT = (W.transpose(-2, -1) > 0).to(W.dtype)
        return W * mask_W * mask_WT
