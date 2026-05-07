"""Type aliases used across the package."""
from __future__ import annotations

import torch

# 2D tensors: (N, d) feature matrix
FeatureTensor = torch.Tensor
# 2D tensors: (N, M) kernel/affinity/attention matrix
GramTensor = torch.Tensor
# 1D or 2D tensors: target values
TargetTensor = torch.Tensor
