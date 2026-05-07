"""Tests for tabkernels.architectures.ft_transformer (Chapter 14)."""
from __future__ import annotations

import pytest
import torch

from tabkernels.architectures.ft_transformer import (
    FeatureTokenizer,
    FTAttentionBlock,
    FTTransformer,
)
from tabkernels.audits._models import StdAttentionDecomposable
from tabkernels.core.base import Architecture, AttentionBlock

# ---------------------------------------------------------------------------
# FeatureTokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_numerical_only_shape():
    tok = FeatureTokenizer(n_num_features=5, cat_cardinalities=[], d_token=16)
    x_num = torch.randn(4, 5)
    out = tok(x_num, None)
    assert out.shape == (4, 5, 16)
    assert tok.n_tokens == 5


def test_tokenizer_categorical_only_shape():
    tok = FeatureTokenizer(n_num_features=0, cat_cardinalities=[3, 4], d_token=8)
    x_cat = torch.tensor([[0, 2], [1, 3], [2, 0]])
    out = tok(None, x_cat)
    assert out.shape == (3, 2, 8)
    assert tok.n_tokens == 2


def test_tokenizer_mixed_shape():
    tok = FeatureTokenizer(n_num_features=2, cat_cardinalities=[5, 3], d_token=12)
    x_num = torch.randn(6, 2)
    x_cat = torch.tensor([[0, 1]] * 6)
    out = tok(x_num, x_cat)
    assert out.shape == (6, 4, 12)


def test_tokenizer_validation_errors():
    with pytest.raises(ValueError):
        FeatureTokenizer(n_num_features=-1, cat_cardinalities=[], d_token=8)
    with pytest.raises(ValueError):
        FeatureTokenizer(n_num_features=2, cat_cardinalities=[], d_token=0)
    with pytest.raises(ValueError):
        FeatureTokenizer(n_num_features=0, cat_cardinalities=[0, 3], d_token=8)
    tok = FeatureTokenizer(n_num_features=2, cat_cardinalities=[3], d_token=8)
    with pytest.raises(ValueError):
        tok(None, torch.tensor([[0]]))  # x_num required
    with pytest.raises(ValueError):
        tok(torch.randn(2, 2), None)  # x_cat required
    with pytest.raises(ValueError):
        tok(torch.randn(2, 7), torch.tensor([[0], [1]]))  # wrong shape
    with pytest.raises(ValueError):
        tok(torch.randn(2, 2), torch.tensor([[0, 1], [1, 2]]))  # wrong cat shape
    empty_tok = FeatureTokenizer(n_num_features=0, cat_cardinalities=[], d_token=8)
    with pytest.raises(ValueError):
        empty_tok(None, None)


# ---------------------------------------------------------------------------
# FTAttentionBlock
# ---------------------------------------------------------------------------


def test_attention_block_subclasses_base():
    block = FTAttentionBlock(d_model=16, n_heads=4, dim_ff=32, dropout=0.0)
    assert isinstance(block, AttentionBlock)


def test_attention_block_forward_shape():
    block = FTAttentionBlock(d_model=16, n_heads=4, dim_ff=32, dropout=0.0)
    x = torch.randn(3, 7, 16)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_attention_block_decompose_shapes():
    block = FTAttentionBlock(d_model=16, n_heads=4, dim_ff=32, dropout=0.0)
    B_S, B_A = block.decompose()
    assert B_S.shape == (16, 16)
    assert B_A.shape == (16, 16)
    # B_S is symmetric, B_A is skew-symmetric.
    assert torch.allclose(B_S, B_S.T, atol=1e-6)
    assert torch.allclose(B_A, -B_A.T, atol=1e-6)


def test_attention_block_energy_split_sums_to_one():
    block = FTAttentionBlock(d_model=16, n_heads=4, dim_ff=32, dropout=0.0)
    a_s, a_a = block.energy_split()
    assert abs(a_s + a_a - 1.0) < 1e-3


def test_attention_block_validates_head_divisor():
    with pytest.raises(ValueError):
        FTAttentionBlock(d_model=15, n_heads=4)


def test_attention_block_predict_not_implemented():
    block = FTAttentionBlock(d_model=16, n_heads=4, dim_ff=32, dropout=0.0)
    with pytest.raises(NotImplementedError):
        block.predict(torch.randn(2, 16), torch.randn(3, 16), torch.randn(3))


# ---------------------------------------------------------------------------
# FTTransformer
# ---------------------------------------------------------------------------


def test_ft_transformer_subclasses_architecture():
    assert issubclass(FTTransformer, Architecture)


def test_ft_transformer_smoke_numerical_only():
    m = FTTransformer(
        n_num_features=5,
        cat_cardinalities=[],
        d_model=16,
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


def test_ft_transformer_smoke_mixed_features():
    m = FTTransformer(
        n_num_features=3,
        cat_cardinalities=[4, 2],
        d_model=16,
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


def test_ft_transformer_default_hyperparams():
    """Default config = Gorishniy 2021 recommended."""
    m = FTTransformer(n_num_features=4)
    assert m.d_model == 192
    assert m.n_heads == 8
    assert m.n_layers == 3
    assert m.dim_ff == 256
    assert abs(m.dropout_p - 0.1) < 1e-9


def test_ft_transformer_validation_errors():
    with pytest.raises(ValueError):
        FTTransformer(n_num_features=4, d_model=15, n_heads=4)
    with pytest.raises(ValueError):
        FTTransformer(n_num_features=4, n_layers=0)
    with pytest.raises(ValueError):
        FTTransformer(n_num_features=4, n_classes=0)
    with pytest.raises(ValueError):
        FTTransformer(n_num_features=0, cat_cardinalities=[])


def test_ft_transformer_attention_blocks_contract():
    """Architecture-contract test from the brief.

    ``attention_blocks()`` returns a non-empty list of ``AttentionBlock``
    instances; each supports ``decompose()`` returning two same-shape tensors.
    """
    m = FTTransformer(
        n_num_features=4,
        d_model=16,
        n_heads=4,
        n_layers=3,
        dim_ff=32,
        n_classes=2,
        dropout=0.0,
    )
    blocks = m.attention_blocks()
    assert len(blocks) == 3
    for b in blocks:
        assert isinstance(b, AttentionBlock)
        B_S, B_A = b.decompose()
        assert B_S.shape == B_A.shape
        assert B_S.shape == (16, 16)


def test_ft_transformer_inner_attention_recognised_by_ch13():
    """The inner StdAttentionDecomposable is recognised by CH13's lens."""
    from tabkernels.transparency import decompose_attention

    m = FTTransformer(
        n_num_features=4,
        d_model=16,
        n_heads=4,
        n_layers=2,
        dim_ff=32,
        dropout=0.0,
    )
    out = decompose_attention(m)
    # 2 layers x 4 heads = 8 recognised heads.
    assert out["B"].shape[0] == 8
    assert out["B_S"].shape == out["B"].shape
    assert out["B_A"].shape == out["B"].shape


def test_ft_transformer_inner_attention_class():
    m = FTTransformer(
        n_num_features=4, d_model=16, n_heads=4, n_layers=1,
        dim_ff=32, dropout=0.0,
    )
    inner = m.layers[0].attn
    assert isinstance(inner, StdAttentionDecomposable)


def test_ft_transformer_train_step_reduces_loss():
    """A short supervised training run should reduce loss.

    Sanity check: gradients flow, the optimizer can fit a linear-ish target.
    """
    torch.manual_seed(0)
    m = FTTransformer(
        n_num_features=4,
        cat_cardinalities=[],
        d_model=32,
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
    for _ in range(40):
        x_num = torch.randn(32, 4)
        y = (x_num @ w > 0).long()
        opt.zero_grad()
        logits = m(x_num)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_ft_transformer_inspect_attention_workflow():
    """End-to-end: inspect_attention runs on an FTTransformer."""
    from tabkernels.transparency import inspect_attention

    torch.manual_seed(0)
    m = FTTransformer(
        n_num_features=4,
        d_model=16,
        n_heads=4,
        n_layers=2,
        dim_ff=32,
        dropout=0.0,
    )
    x_num = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    report = inspect_attention(m, x_num, y=y)
    assert "alpha_S" in report
    assert "alpha_A" in report
    assert abs(report["alpha_S"] + report["alpha_A"] - 1.0) < 1e-3
    # 2 layers x 4 heads = 8 recognised heads.
    assert len(report["per_head"]) == 8
