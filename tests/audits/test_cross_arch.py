"""Tests for tabkernels.audits.cross_arch (Chapter 18)."""
from __future__ import annotations

import torch

from tabkernels.architectures import TabICLLite, TabPFNLite
from tabkernels.audits import run_cross_arch_audit


def _quick_train(model, n_steps=20):
    """A few SGD steps so B = W_Q^T W_K is non-trivial."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(n_steps):
        X_ctx = torch.randn(16, 4); X_q = torch.randn(8, 4)
        w = torch.randn(4)
        y_ctx = X_ctx @ w + 0.1 * torch.randn(16)
        y_q = X_q @ w + 0.1 * torch.randn(8)
        opt.zero_grad()
        loss = ((model(X_q, X_ctx, y_ctx) - y_q) ** 2).mean()
        loss.backward(); opt.step()
    return model


def test_audit_runs_on_two_archs():
    torch.manual_seed(0)
    pfn = _quick_train(TabPFNLite(d_in=4, d_model=16, n_heads=2, n_layers=2))
    icl = _quick_train(TabICLLite(d_in=4, d_model=16, n_heads=2, n_layers=2))
    report = run_cross_arch_audit({"TabPFN": pfn, "TabICL": icl})
    summary = report["metrics"]["summary"]
    assert {row["architecture"] for row in summary} == {"TabPFN", "TabICL"}
    for row in summary:
        assert row["n_blocks"] > 0
        assert 0 <= row["mean_alpha_S"] <= 1.0
        assert 0 <= row["mean_alpha_A"] <= 1.0


def test_audit_with_target_B_reports_cosine():
    torch.manual_seed(0)
    pfn = _quick_train(TabPFNLite(d_in=4, d_model=16, n_heads=2, n_layers=1))
    target_B = torch.randn(16, 16)
    target_B = target_B + target_B.T  # symmetric target
    report = run_cross_arch_audit({"TabPFN": pfn}, target_B=target_B)
    summary = report["metrics"]["summary"]
    assert summary[0]["mean_cos_sym"] != float("nan")


def test_audit_skips_targetless_recovery():
    torch.manual_seed(0)
    pfn = _quick_train(TabPFNLite(d_in=4, d_model=16, n_heads=2, n_layers=1))
    report = run_cross_arch_audit({"TabPFN": pfn}, target_B=None)
    per_arch = report["per_seed"][0]
    assert per_arch["recovery"] == []


def test_audit_persists_to_json(tmp_path):
    from tabkernels.audits.base import load_report
    torch.manual_seed(0)
    pfn = _quick_train(TabPFNLite(d_in=4, d_model=16, n_heads=2, n_layers=1))
    out = tmp_path / "cross_arch.json"
    report = run_cross_arch_audit({"TabPFN": pfn}, save_to=out)
    loaded = load_report(out)
    assert loaded["config"]["kind"] == "cross_arch_audit"
    assert "TabPFN" in loaded["config"]["architectures"]
