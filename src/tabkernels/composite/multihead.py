"""Multi-head kernel composition (Chapter 9 §9.1-§9.2)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from tabkernels.core.base import Kernel


class MultiHead(Kernel):
    """Average of per-head kernels.

    Two modes:
      - chunk: the same kernel applied to chunked feature vectors.
      - mkl: separate kernel instances; their outputs are averaged.

    Parameters
    ----------
    kernels : sequence of Kernel
        One kernel per head. They should accept the same input shape.
    """

    def __init__(self, kernels: Sequence[Kernel]):
        super().__init__()
        if len(kernels) < 1:
            raise ValueError("at least one kernel required")
        self.heads = nn.ModuleList(kernels)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(X1, X2) for h in self.heads]).mean(dim=0)


class ChunkMultiHead(Kernel):
    """Same kernel applied to equal-size chunks of the input.

    Splits each input vector into ``H`` chunks of size d/H and applies the same
    kernel to each chunk.

    Parameters
    ----------
    kernel_factory : callable
        Function (d_chunk: int) -> Kernel; called once per head.
    H : int
        Number of heads.
    d_in : int
        Total input dimension.
    """

    def __init__(self, kernel_factory, H: int, d_in: int):
        super().__init__()
        if d_in % H != 0:
            raise ValueError(f"d_in={d_in} not divisible by H={H}")
        self.H = H
        self.d_chunk = d_in // H
        self.heads = nn.ModuleList(
            [kernel_factory(self.d_chunk) for _ in range(H)]
        )

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        chunks1 = X1.chunk(self.H, dim=-1)
        chunks2 = X2.chunk(self.H, dim=-1)
        return torch.stack(
            [self.heads[h](chunks1[h], chunks2[h]) for h in range(self.H)]
        ).mean(dim=0)
