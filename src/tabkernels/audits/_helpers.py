"""Private training / evaluation helpers shared by the audit modules.

These wrap the ``train_*`` / ``eval_*`` loops that the original ``~/jupyterlab/ICL``
scripts kept inline. Like ``_models``, these are placeholders for what
AGENT-CH21 (training/evaluation API) will replace.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from ._models import KernelAttention


def get_device() -> torch.device:
    """The CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# KernelAttention training / eval (Oracle, Directional).
# ----------------------------------------------------------------------------


def train_kernel_attention(model: KernelAttention, X_train: np.ndarray,
                           y_train: np.ndarray, *, task: str = "reg",
                           epochs: int = 1500, lr: float = 3e-2,
                           weight_decay: float = 1e-4,
                           causal: bool = False,
                           order_train: np.ndarray | None = None
                           ) -> KernelAttention:
    """Fit a :class:`KernelAttention` block by leave-one-out attention loss."""
    device = get_device()
    model.to(device)
    X = torch.tensor(X_train, device=device)
    y = torch.tensor(y_train, device=device)
    order_t = (torch.tensor(order_train.astype(np.float32), device=device)
               if order_train is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    for _ in range(epochs):
        y_pred = model.predict(
            X, X, y,
            order_q=order_t, order_t=order_t,
            causal=causal, mask_diag=True,
        )
        if task == "reg":
            loss = F.mse_loss(y_pred, y)
        else:
            y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def eval_kernel_attention(model: KernelAttention, X_train: np.ndarray,
                          y_train: np.ndarray, X_test: np.ndarray,
                          y_test: np.ndarray, *, task: str = "reg",
                          causal: bool = False,
                          order_train: np.ndarray | None = None,
                          order_test: np.ndarray | None = None) -> dict:
    """Evaluate a :class:`KernelAttention` block on held-out data."""
    device = get_device()
    model.eval()
    X_tr = torch.tensor(X_train, device=device)
    y_tr = torch.tensor(y_train, device=device)
    X_te = torch.tensor(X_test, device=device)
    y_te = torch.tensor(y_test, device=device)
    order_tr_t = (torch.tensor(order_train.astype(np.float32), device=device)
                  if order_train is not None else None)
    order_te_t = (torch.tensor(order_test.astype(np.float32), device=device)
                  if order_test is not None else None)
    with torch.no_grad():
        y_pred = model.predict(
            X_te, X_tr, y_tr,
            order_q=order_te_t, order_t=order_tr_t,
            causal=causal, mask_diag=False,
        )
        if task == "reg":
            return {"mse": F.mse_loss(y_pred, y_te).item()}
        y_pred_bin = (y_pred > 0.5).float()
        return {"acc": (y_pred_bin == y_te).float().mean().item()}


# ----------------------------------------------------------------------------
# Pairwise predictor training / eval.
# ----------------------------------------------------------------------------


def train_pairwise(model: nn.Module, Z: np.ndarray, R: np.ndarray,
                   train_mask: np.ndarray, *, epochs: int = 2000,
                   lr: float = 3e-2, weight_decay: float = 1e-5) -> nn.Module:
    """Fit a :class:`PairwisePredictor` on the masked entries of ``R``."""
    device = get_device()
    model.to(device)
    Zt = torch.tensor(Z, device=device)
    Rt = torch.tensor(R, device=device)
    mask_tr = torch.tensor(train_mask, device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    for _ in range(epochs):
        Rhat = model(Zt)
        loss = ((Rhat - Rt) ** 2 * mask_tr).sum() / (mask_tr.sum() + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def eval_pairwise(model: nn.Module, Z: np.ndarray, R: np.ndarray,
                  test_mask: np.ndarray) -> float:
    """Mean-squared error of a fitted :class:`PairwisePredictor` on masked test entries."""
    device = get_device()
    model.eval()
    Zt = torch.tensor(Z, device=device)
    Rt = torch.tensor(R, device=device)
    mask_te = torch.tensor(test_mask, device=device, dtype=torch.float32)
    with torch.no_grad():
        Rhat = model(Zt)
        return float(((Rhat - Rt) ** 2 * mask_te).sum() / (mask_te.sum() + 1e-8))


# ----------------------------------------------------------------------------
# Numpy decomposition helpers (used by structure analysis + pairwise diagnostics).
# ----------------------------------------------------------------------------


def cosine_F_np(A: np.ndarray, B: np.ndarray) -> float:
    a, b = A.flatten(), B.flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def energy_split_np(M: np.ndarray) -> tuple[float, float]:
    """``(sym_frac, asym_frac)`` Frobenius energy split of a square matrix."""
    n = (M ** 2).sum() + 1e-12
    M_S = (M + M.T) / 2
    M_A = (M - M.T) / 2
    return float((M_S ** 2).sum() / n), float((M_A ** 2).sum() / n)


# ----------------------------------------------------------------------------
# Tiny synthetic ICL prior used by the migrated ICL-audit smoke tests.
# ----------------------------------------------------------------------------


def make_synthetic_icl_episode(B: int, n_ctx: int, n_q: int, *,
                               d: int = 8, n_classes: int = 2,
                               seed: int | None = None,
                               device: torch.device | None = None
                               ) -> tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor]:
    """Cheap stand-in for the ARF+MLP-SCM prior used in the original scripts.

    The full ARF prior depends on a real-data corpus and several thousand
    training epochs; for the migrated audits we expose this lightweight
    Gaussian + linear-threshold prior so smoke tests can run in a few seconds
    on CPU. Audits can swap in a richer prior via the ``prior`` argument once
    AGENT-CH21 lands.
    """
    if device is None:
        device = get_device()
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    n_tot = n_ctx + n_q
    X_all = torch.randn(B, n_tot, d, generator=g)
    # Per-batch random hyperplane decision rule.
    w = torch.randn(B, d, generator=g)
    score = torch.einsum("bnd,bd->bn", X_all, w)
    thresh = score.median(dim=-1, keepdim=True).values
    y_all = (score > thresh).long()
    if n_classes != 2:
        # Random multi-class projection — only used if a caller asks for it.
        proj = torch.randn(B, d, n_classes, generator=g)
        logits = torch.einsum("bnd,bdc->bnc", X_all, proj)
        y_all = logits.argmax(dim=-1)
    X_all = X_all.to(device)
    y_all = y_all.to(device)
    return (X_all[:, :n_ctx], y_all[:, :n_ctx],
            X_all[:, n_ctx:], y_all[:, n_ctx:])


# ----------------------------------------------------------------------------
# ICL training / eval (lightweight reimplementation of the routines in
# ``symmetric_kernel_icl_arf.py``, restricted to the synthetic Gaussian prior
# above). Audits that require the full ARF pool can override the ``sample_fn``.
# ----------------------------------------------------------------------------


def _model_tokenizer(model: nn.Module) -> nn.Module:
    """Locate the :class:`TabularTokenizer` on a model (handles boost wrappers)."""
    if hasattr(model, "tokenizer"):
        return cast(nn.Module, model.tokenizer)
    if hasattr(model, "delta"):
        delta = cast(nn.Module, model.delta)
        if hasattr(delta, "tokenizer"):
            return cast(nn.Module, delta.tokenizer)
    raise AttributeError(f"cannot locate tokenizer on {type(model).__name__}")


def _model_in_features(model: nn.Module) -> int:
    """Return the input feature dimension of a tokenized ICL transformer."""
    tok = _model_tokenizer(model)
    feat_proj = cast(nn.Linear, tok.feat_proj)
    return int(feat_proj.in_features)


def _model_n_classes(model: nn.Module) -> int:
    return int(cast(int, model.n_classes))


def train_icl_synthetic(model: nn.Module, *, n_steps: int = 200,
                        batch_size: int = 16, n_ctx_range: tuple = (4, 32),
                        n_q: int = 8, lr: float = 3e-4,
                        seed: int = 0) -> list[float]:
    """Train an ICL transformer on the synthetic Gaussian + threshold prior."""
    device = get_device()
    model.to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps, 1))
    losses: list[float] = []
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    d_in = _model_in_features(model)
    n_classes = _model_n_classes(model)
    for step in range(n_steps):
        n_ctx = int(torch.randint(n_ctx_range[0], n_ctx_range[1] + 1, (1,),
                                  generator=g).item())
        Xc, yc, Xq, yq = make_synthetic_icl_episode(
            B=batch_size, n_ctx=n_ctx, n_q=n_q,
            d=d_in,
            n_classes=n_classes,
            seed=seed * 100003 + step,
            device=device,
        )
        logits = model(Xc, yc, Xq)
        loss = F.cross_entropy(
            logits.reshape(-1, n_classes), yq.reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
    return losses


@torch.no_grad()
def evaluate_icl_synthetic(model: nn.Module, *, n_episodes: int = 20,
                           n_ctx: int = 16, n_q: int = 32,
                           seed: int = 7919) -> dict:
    """Held-out accuracy + label-swap delta + simple context-size scaling."""
    device = get_device()
    model.eval()
    d = _model_in_features(model)
    n_classes = _model_n_classes(model)

    # Held-out accuracy.
    accs: list[float] = []
    for i in range(max(1, n_episodes // 5)):
        Xc, yc, Xq, yq = make_synthetic_icl_episode(
            B=5, n_ctx=n_ctx, n_q=n_q, d=d, n_classes=n_classes,
            seed=seed + i, device=device,
        )
        logits = model(Xc, yc, Xq)
        accs.append(float((logits.argmax(-1) == yq).float().mean().item()))
    holdout = float(np.mean(accs))

    # Label swap delta.
    swaps: list[float] = []
    for i in range(max(1, n_episodes // 5)):
        Xc, yc, Xq, yq = make_synthetic_icl_episode(
            B=1, n_ctx=n_ctx, n_q=n_q, d=d, n_classes=n_classes,
            seed=seed + 1000 + i, device=device,
        )
        a_true = float((model(Xc, yc, Xq).argmax(-1) == yq).float().mean().item())
        a_swap = float(
            (model(Xc, 1 - yc, Xq).argmax(-1) == yq).float().mean().item()
        )
        swaps.append(a_true - a_swap)
    swap = float(np.mean(swaps))

    # Context-size scaling at a small grid.
    scaling: list[tuple[int, float]] = []
    for n in (max(2, n_ctx // 4), n_ctx, n_ctx * 2):
        scs: list[float] = []
        for i in range(max(1, n_episodes // 5)):
            Xc, yc, Xq, yq = make_synthetic_icl_episode(
                B=2, n_ctx=n, n_q=n_q, d=d, n_classes=n_classes,
                seed=seed + 2000 + i, device=device,
            )
            scs.append(
                float((model(Xc, yc, Xq).argmax(-1) == yq).float().mean().item())
            )
        scaling.append((n, float(np.mean(scs))))

    return {"holdout": holdout, "swap": swap, "scaling": scaling}


# Suppress lint warnings about unused symbols when imported only for re-export.
_ = math
