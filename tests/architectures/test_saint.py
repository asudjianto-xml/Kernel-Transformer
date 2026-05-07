"""Tests for tabkernels.architectures.saint (Chapter 15)."""
from __future__ import annotations

import pytest
import torch

from tabkernels.architectures.saint import (
    SAINT,
    SAINTAttentionBlock,
    SAINTFeatureTokenizer,
)
from tabkernels.audits._models import StdAttentionDecomposable
from tabkernels.core.base import Architecture, AttentionBlock

# ---------------------------------------------------------------------------
# SAINTFeatureTokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_numerical_only_shape():
    tok = SAINTFeatureTokenizer(n_num_features=5, cat_cardinalities=[], d_token=16)
    x_num = torch.randn(4, 5)
    out = tok(x_num, None)
    assert out.shape == (4, 5, 16)
    assert tok.n_tokens == 5


def test_tokenizer_categorical_only_shape():
    tok = SAINTFeatureTokenizer(n_num_features=0, cat_cardinalities=[3, 4], d_token=8)
    x_cat = torch.tensor([[0, 2], [1, 3], [2, 0]])
    out = tok(None, x_cat)
    assert out.shape == (3, 2, 8)
    assert tok.n_tokens == 2


def test_tokenizer_mixed_shape():
    tok = SAINTFeatureTokenizer(n_num_features=2, cat_cardinalities=[5, 3], d_token=12)
    x_num = torch.randn(6, 2)
    x_cat = torch.tensor([[0, 1]] * 6)
    out = tok(x_num, x_cat)
    assert out.shape == (6, 4, 12)


def test_tokenizer_validation_errors():
    with pytest.raises(ValueError):
        SAINTFeatureTokenizer(n_num_features=-1, cat_cardinalities=[], d_token=8)
    with pytest.raises(ValueError):
        SAINTFeatureTokenizer(n_num_features=2, cat_cardinalities=[], d_token=0)
    with pytest.raises(ValueError):
        SAINTFeatureTokenizer(n_num_features=0, cat_cardinalities=[0, 3], d_token=8)
    tok = SAINTFeatureTokenizer(n_num_features=2, cat_cardinalities=[3], d_token=8)
    with pytest.raises(ValueError):
        tok(None, torch.tensor([[0]]))
    with pytest.raises(ValueError):
        tok(torch.randn(2, 2), None)
    with pytest.raises(ValueError):
        tok(torch.randn(2, 7), torch.tensor([[0], [1]]))
    with pytest.raises(ValueError):
        tok(torch.randn(2, 2), torch.tensor([[0, 1], [1, 2]]))
    empty_tok = SAINTFeatureTokenizer(n_num_features=0, cat_cardinalities=[], d_token=8)
    with pytest.raises(ValueError):
        empty_tok(None, None)


# ---------------------------------------------------------------------------
# SAINTAttentionBlock
# ---------------------------------------------------------------------------


def test_attention_block_subclasses_base():
    block = SAINTAttentionBlock(
        d_model=16, n_heads=4, dim_ff=32, dropout=0.0, axis="column"
    )
    assert isinstance(block, AttentionBlock)


def test_column_block_forward_shape():
    block = SAINTAttentionBlock(
        d_model=16, n_heads=4, dim_ff=32, dropout=0.0, axis="column"
    )
    x = torch.randn(3, 7, 16)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_row_block_forward_shape():
    # Row block: d_model = F * d_token; attends across batch axis.
    F, d_token, B = 5, 4, 6
    block = SAINTAttentionBlock(
        d_model=F * d_token, n_heads=4, dim_ff=32, dropout=0.0, axis="row"
    )
    x = torch.randn(B, F, d_token)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_column_block_decompose_shape():
    block = SAINTAttentionBlock(
        d_model=16, n_heads=4, dim_ff=32, dropout=0.0, axis="column"
    )
    B_S, B_A = block.decompose()
    assert B_S.shape == (16, 16)
    assert B_A.shape == (16, 16)
    assert torch.allclose(B_S, B_S.T, atol=1e-6)
    assert torch.allclose(B_A, -B_A.T, atol=1e-6)


def test_row_block_decompose_shape():
    F, d_token = 5, 4
    block = SAINTAttentionBlock(
        d_model=F * d_token, n_heads=4, dim_ff=32, dropout=0.0, axis="row"
    )
    B_S, B_A = block.decompose()
    assert B_S.shape == (F * d_token, F * d_token)
    assert B_A.shape == (F * d_token, F * d_token)
    assert torch.allclose(B_S, B_S.T, atol=1e-6)
    assert torch.allclose(B_A, -B_A.T, atol=1e-6)


def test_attention_block_energy_split_sums_to_one():
    block = SAINTAttentionBlock(
        d_model=16, n_heads=4, dim_ff=32, dropout=0.0, axis="column"
    )
    a_s, a_a = block.energy_split()
    assert abs(a_s + a_a - 1.0) < 1e-3


def test_attention_block_validates_head_divisor():
    with pytest.raises(ValueError):
        SAINTAttentionBlock(d_model=15, n_heads=4, axis="column")


def test_attention_block_validates_axis():
    with pytest.raises(ValueError):
        SAINTAttentionBlock(d_model=16, n_heads=4, axis="diagonal")


def test_attention_block_predict_not_implemented():
    block = SAINTAttentionBlock(d_model=16, n_heads=4, axis="column")
    with pytest.raises(NotImplementedError):
        block.predict(torch.randn(2, 16), torch.randn(3, 16), torch.randn(3))


def test_attention_block_carries_axis_attribute():
    col = SAINTAttentionBlock(d_model=16, n_heads=4, axis="column")
    row = SAINTAttentionBlock(d_model=16, n_heads=4, axis="row")
    assert col.axis == "column"
    assert row.axis == "row"


# ---------------------------------------------------------------------------
# SAINT
# ---------------------------------------------------------------------------


def test_saint_subclasses_architecture():
    assert issubclass(SAINT, Architecture)


def test_saint_smoke_numerical_only():
    m = SAINT(
        n_num_features=5,
        cat_cardinalities=[],
        n_samples=4,
        d_token=16,
        n_heads=4,
        n_layers=2,
        dim_ff=32,
        n_classes=3,
        dropout=0.0,
    )
    x_num = torch.randn(4, 5)
    out = m(x_num)
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_saint_smoke_mixed_features():
    m = SAINT(
        n_num_features=3,
        cat_cardinalities=[4, 2],
        n_samples=5,
        d_token=16,
        n_heads=4,
        n_layers=2,
        dim_ff=32,
        n_classes=2,
        dropout=0.0,
    )
    x_num = torch.randn(5, 3)
    x_cat = torch.tensor([[0, 0], [1, 1], [2, 0], [3, 1], [0, 0]])
    out = m(x_num, x_cat)
    assert out.shape == (5, 2)


def test_saint_validation_errors():
    with pytest.raises(ValueError):
        SAINT(n_num_features=4, n_samples=4, d_token=15, n_heads=4)
    with pytest.raises(ValueError):
        SAINT(n_num_features=4, n_samples=4, n_layers=0)
    with pytest.raises(ValueError):
        SAINT(n_num_features=4, n_samples=4, n_classes=0)
    with pytest.raises(ValueError):
        SAINT(n_num_features=4, n_samples=0)
    with pytest.raises(ValueError):
        SAINT(n_num_features=0, cat_cardinalities=[], n_samples=4)
    # Row width F*d_token must divide n_heads.
    with pytest.raises(ValueError):
        SAINT(
            n_num_features=3,
            cat_cardinalities=[],
            n_samples=4,
            d_token=8,
            n_heads=5,  # 3*8=24 not divisible by 5
        )


def test_saint_attention_blocks_contract():
    """Architecture contract: ``attention_blocks()`` returns 2*n_layers blocks
    (one column + one row per layer), each supporting ``decompose()``.
    """
    m = SAINT(
        n_num_features=4,
        cat_cardinalities=[],
        n_samples=8,
        d_token=16,
        n_heads=4,
        n_layers=3,
        dim_ff=32,
        n_classes=2,
        dropout=0.0,
    )
    blocks = m.attention_blocks()
    # 3 layers x (column + row) = 6 blocks.
    assert len(blocks) == 6
    for b in blocks:
        assert isinstance(b, AttentionBlock)
        B_S, B_A = b.decompose()
        assert B_S.shape == B_A.shape


def test_saint_separates_row_and_column_blocks():
    """``column_attention_blocks()`` and ``row_attention_blocks()`` each
    return ``n_layers`` blocks with the right axis attribute.
    """
    n_layers = 3
    F_total, d_token = 4, 16
    m = SAINT(
        n_num_features=F_total,
        cat_cardinalities=[],
        n_samples=8,
        d_token=d_token,
        n_heads=4,
        n_layers=n_layers,
        dim_ff=32,
        dropout=0.0,
    )
    cols = m.column_attention_blocks()
    rows = m.row_attention_blocks()
    assert len(cols) == n_layers
    assert len(rows) == n_layers
    for b in cols:
        assert b.axis == "column"
        B_S, _ = b.decompose()
        assert B_S.shape == (d_token, d_token)
    for b in rows:
        assert b.axis == "row"
        B_S, _ = b.decompose()
        assert B_S.shape == (F_total * d_token, F_total * d_token)


def test_saint_inner_attention_class():
    m = SAINT(
        n_num_features=4,
        cat_cardinalities=[],
        n_samples=4,
        d_token=16,
        n_heads=4,
        n_layers=1,
        dim_ff=32,
        dropout=0.0,
    )
    assert isinstance(m.column_layers[0].attn, StdAttentionDecomposable)
    assert isinstance(m.row_layers[0].attn, StdAttentionDecomposable)


def test_saint_train_step_reduces_loss():
    """Short training run reduces loss."""
    torch.manual_seed(0)
    m = SAINT(
        n_num_features=4,
        cat_cardinalities=[],
        n_samples=32,
        d_token=16,
        n_heads=4,
        n_layers=2,
        dim_ff=32,
        n_classes=2,
        dropout=0.0,
    )
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    w = torch.randn(4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(30):
        x_num = torch.randn(32, 4)
        y = (x_num @ w > 0).long()
        opt.zero_grad()
        logits = m(x_num)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_saint_inspect_attention_workflow():
    """End-to-end: inspect_attention runs on a SAINT model.

    SAINT has 2*n_layers attention blocks; each block contains a
    StdAttentionDecomposable that the Ch 13 lens recognises.
    """
    from tabkernels.transparency import inspect_attention

    torch.manual_seed(0)
    m = SAINT(
        n_num_features=4,
        cat_cardinalities=[],
        n_samples=8,
        d_token=16,
        n_heads=4,
        n_layers=2,
        dim_ff=32,
        n_classes=2,
        dropout=0.0,
    )
    x_num = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    report = inspect_attention(m, x_num, y=y)
    assert "alpha_S" in report
    assert "alpha_A" in report
    assert abs(report["alpha_S"] + report["alpha_A"] - 1.0) < 1e-3
    # Per-axis inspection: SAINT's distinguishing capability.
    # Both column and row blocks are visited by attention_blocks().
    assert len(report["per_head"]) > 0
