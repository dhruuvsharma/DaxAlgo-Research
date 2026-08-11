from __future__ import annotations

from dataclasses import replace

import torch

from daxalgo_numeric_transformer.config import ModelConfig
from daxalgo_numeric_transformer.masks import masked_softmax
from daxalgo_numeric_transformer.model.blocks import PreNormTransformerBlock
from daxalgo_numeric_transformer.model.numeric_transformer import NumericMarketTransformer
from daxalgo_numeric_transformer.model.pooling import CausalPatchPool
from daxalgo_numeric_transformer.synthetic import make_synthetic_batch


def test_temporal_causal_mask_has_zero_future_value_gradient() -> None:
    token_count = 6
    scores = torch.randn(1, token_count, token_count, generator=torch.Generator().manual_seed(13))
    causal_valid = torch.ones(token_count, token_count, dtype=torch.bool).tril().unsqueeze(0)
    values = torch.randn(
        1,
        token_count,
        3,
        generator=torch.Generator().manual_seed(17),
        requires_grad=True,
    )

    weights = masked_softmax(scores, causal_valid, dim=-1)
    encoded = weights @ values
    encoded[:, 2].sum().backward()

    assert values.grad is not None
    assert values.grad[:, :3].abs().sum() > 0
    assert values.grad[:, 3:].count_nonzero() == 0


def test_actual_temporal_block_has_zero_future_input_gradient() -> None:
    torch.manual_seed(19)
    block = PreNormTransformerBlock(
        d_model=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        causal=True,
    ).eval()
    values = torch.randn(1, 6, 8, requires_grad=True)
    valid = torch.ones(1, 6, dtype=torch.bool)

    encoded = block(values, valid)
    encoded[:, 2].sum().backward()

    assert values.grad is not None
    assert values.grad[:, :3].abs().sum() > 0
    assert values.grad[:, 3:].count_nonzero() == 0


def test_patch_pool_uses_only_frames_inside_each_non_overlapping_patch() -> None:
    torch.manual_seed(23)
    pool = CausalPatchPool(d_model=8, hidden_dim=4, patch_width=4).eval()
    frames = torch.randn(1, 8, 8, requires_grad=True)
    valid = torch.ones(1, 8, dtype=torch.bool)

    pooled, patch_mask = pool(frames, valid)

    assert pooled.shape == (1, 2, 8)
    assert patch_mask.tolist() == [[True, True]]
    pooled[:, 0].sum().backward()
    assert frames.grad is not None
    assert frames.grad[:, :4].abs().sum() > 0
    assert frames.grad[:, 4:].count_nonzero() == 0


def test_all_masked_patch_is_zero_and_invalid() -> None:
    torch.manual_seed(29)
    pool = CausalPatchPool(d_model=8, hidden_dim=4, patch_width=4).eval()
    frames = torch.randn(1, 8, 8)
    valid = torch.tensor([[True, True, True, True, False, False, False, False]])

    pooled, patch_mask = pool(frames, valid)

    assert patch_mask.tolist() == [[True, False]]
    assert torch.isfinite(pooled).all()
    assert pooled[:, 1].count_nonzero() == 0


def test_l2_level_attention_never_crosses_snapshot_rows() -> None:
    torch.manual_seed(31)
    config = ModelConfig.tiny()
    model = NumericMarketTransformer(config).eval()
    batch = make_synthetic_batch(config, batch_size=1, seed=37, l2_mode="present")
    valid_snapshots = batch.l2_snapshot_mask[0].nonzero(as_tuple=False).flatten()
    reference_index = int(valid_snapshots[0])
    changed_index = int(valid_snapshots[1])
    captures: list[torch.Tensor] = []

    handle = model.level_blocks[-1].register_forward_hook(
        lambda _module, _inputs, output: captures.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            model.encode_modalities(batch)
        baseline = captures.pop()
        changed_l2 = batch.l2.clone()
        changed_l2[:, changed_index] += 100.0 * batch.l2_level_mask[
            :, changed_index
        ].unsqueeze(-1)
        with torch.no_grad():
            model.encode_modalities(replace(batch, l2=changed_l2))
        changed = captures.pop()
    finally:
        handle.remove()

    torch.testing.assert_close(
        baseline[reference_index], changed[reference_index], rtol=0, atol=0
    )
    assert not torch.equal(baseline[changed_index], changed[changed_index])

