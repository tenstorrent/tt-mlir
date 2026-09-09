# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tilize / untilize: LazyTensor-only inputs, dtype constants, dtype
override f32 -> bf16 -> f32 round trip."""

import torch
import d2m_jit as d2m


def make_layout(tiled=True):
    return d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
        tiled=tiled,
    )


def test_dtype_constants_exist():
    assert d2m.float32 is not None
    assert d2m.float16 is not None
    assert d2m.bfloat16 is not None


def test_tilize_rejects_torch_tensor():
    t = torch.zeros(64, 64)
    try:
        d2m.tilize(t)
    except TypeError:
        return
    raise AssertionError("tilize on torch.Tensor did not raise")


def test_untilize_rejects_torch_tensor():
    t = torch.zeros(64, 64)
    try:
        d2m.untilize(t)
    except TypeError:
        return
    raise AssertionError("untilize on torch.Tensor did not raise")


def test_dtype_round_trip():
    """f32 -> untilize(dtype=bf16) -> tilize(dtype=f32) -> to_host."""
    t = torch.randn(64, 64, dtype=torch.float32)
    L = make_layout()
    lt = d2m.to_layout(t, L)
    lt_bf = d2m.untilize(lt, dtype=d2m.bfloat16)
    assert lt_bf.layout.dtype != lt.layout.dtype
    assert lt_bf.layout.tiled is False
    lt_back = d2m.tilize(lt_bf, dtype=d2m.float32)
    assert lt_back.layout.tiled is True
    out = lt_back.to_host()
    assert torch.allclose(t, out, atol=0.05), "f32 -> bf16 -> f32 mismatch"
