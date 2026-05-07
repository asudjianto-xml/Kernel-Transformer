"""Common evaluation metrics used across audits and chapters."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Mean squared error."""
    return F.mse_loss(y_pred, y_true).item()


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor,
             threshold: float = 0.5) -> float:
    """Binary accuracy. y_pred is treated as probability if continuous."""
    if y_pred.dtype.is_floating_point and y_pred.dim() == y_true.dim():
        y_pred_bin = (y_pred > threshold).to(y_true.dtype)
    else:
        y_pred_bin = y_pred.to(y_true.dtype)
    return (y_pred_bin == y_true).float().mean().item()


def r_squared(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Coefficient of determination R^2."""
    ss_res = ((y_pred - y_true) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().item() + 1e-12
    return 1 - ss_res / ss_tot


def label_swap_delta(model: torch.nn.Module,
                     X_ctx: torch.Tensor, y_ctx: torch.Tensor,
                     X_q: torch.Tensor) -> float:
    """Sensitivity-to-context-labels diagnostic (used in Ch 10).

    Compares the model's prediction with original y_ctx vs flipped y_ctx;
    returns the average change in predicted-class confidence direction.

    Specifically used in the ICL audit (Section 10.5).
    """
    with torch.no_grad():
        p_orig = model(X_ctx, y_ctx, X_q)
        p_flip = model(X_ctx, 1 - y_ctx, X_q)
    return (p_orig - p_flip).mean().item()
