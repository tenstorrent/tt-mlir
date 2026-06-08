# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Bespoke-signature tile ops: clamp_scalar, typecast, tile_transpose.

Each carries one or more Python-literal arguments that lower to MLIR
attributes on the underlying `d2m.tile_*` op (rather than to a runtime
operand). These tests exercise the silicon lowering with a minimal
single-tile-per-shard layout."""

import torch
import d2m_jit as d2m


def _make_layout(dtype=d2m.float32, shape=(64, 64)):
    return d2m.Layout(shape=shape, dtype=dtype, block_shape=[1, 1], grid_shape=[2, 2])


# ---------------------------------------------------------------------------
# clamp_scalar
# ---------------------------------------------------------------------------


@d2m.kernel
def k_clamp(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            r = clamp_scalar(x, -0.5, 0.5)
            remote_store(out_t, [m_off + m, n_off + n], r)


def test_clamp_scalar_float():
    """clamp_scalar(x, -0.5, 0.5) clips a normal distribution to [-0.5, 0.5]."""
    t = torch.randn(64, 64, dtype=torch.float32)
    L = _make_layout()
    out = d2m.empty(L)
    k_clamp(d2m.to_layout(t, L), out, 1, 1, grid=(2, 2))
    result = out.to_host()
    expected = t.clamp(-0.5, 0.5)
    diff = (expected - result).abs().max().item()
    assert diff < 0.01, f"clamp_scalar max diff {diff}"


# Asymmetric bounds covered by the free-function form (method-style is
# unavailable: see the note next to TensorBlock.tile_transpose in api.py).
@d2m.kernel
def k_clamp_asymmetric(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            r = clamp_scalar(x, 0.0, 2.0)
            remote_store(out_t, [m_off + m, n_off + n], r)


def test_clamp_scalar_asymmetric():
    """Asymmetric bounds [0.0, 2.0]: covers the min!=−max code path."""
    t = torch.randn(64, 64, dtype=torch.float32) * 2.0
    L = _make_layout()
    out = d2m.empty(L)
    k_clamp_asymmetric(d2m.to_layout(t, L), out, 1, 1, grid=(2, 2))
    result = out.to_host()
    expected = t.clamp(0.0, 2.0)
    diff = (expected - result).abs().max().item()
    assert diff < 0.01, f"clamp_scalar asymmetric max diff {diff}"


# ---------------------------------------------------------------------------
# typecast
# ---------------------------------------------------------------------------


@d2m.kernel
def k_typecast(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            r = typecast(x, "bf16")
            remote_store(out_t, [m_off + m, n_off + n], r)


def test_typecast_f32_to_bf16():
    """f32 -> bf16 mid-kernel via tile_typecast. Numerical correctness
    relies on the pipeline running `convert-d2m-to-ttmetal` while
    `ttkernel.typecast_tile` is still present (so the per-thread
    UnpackToDestMode picks Fp32 rather than Default) -- see the comment
    on `_pipeline_passes` in tools/d2m-jit/_src/builder.py."""
    t = torch.randn(64, 64, dtype=torch.float32)
    L_in = _make_layout(dtype=d2m.float32)
    L_out = _make_layout(dtype=d2m.bfloat16)
    out = d2m.empty(L_out)
    k_typecast(d2m.to_layout(t, L_in), out, 1, 1, grid=(2, 2))
    result = out.to_host()
    assert result.dtype == torch.bfloat16
    expected = t.to(torch.bfloat16)
    diff = (expected.float() - result.float()).abs().max().item()
    # bf16 truncation rounding -- one ULP at this magnitude is ~0.016.
    assert diff < 0.05, f"typecast f32->bf16 max diff {diff}"


# ---------------------------------------------------------------------------
# tile_transpose
# ---------------------------------------------------------------------------


@d2m.kernel
def k_tile_transpose(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            r = tile_transpose(x)
            remote_store(out_t, [m_off + m, n_off + n], r)


def test_tile_transpose_per_tile():
    """tile_transpose transposes each 32x32 tile in place. With one tile
    per shard, the output's [cy*32:(cy+1)*32, cx*32:(cx+1)*32] block is
    the transpose of the corresponding input block (NOT a logical
    transpose of the full 64x64 tensor)."""
    t = torch.randn(64, 64, dtype=torch.float32)
    L = _make_layout()
    out = d2m.empty(L)
    k_tile_transpose(d2m.to_layout(t, L), out, 1, 1, grid=(2, 2))
    result = out.to_host()

    expected = torch.empty_like(t)
    for cy in range(2):
        for cx in range(2):
            r0, r1 = cy * 32, (cy + 1) * 32
            c0, c1 = cx * 32, (cx + 1) * 32
            expected[r0:r1, c0:c1] = t[r0:r1, c0:c1].T

    diff = (expected - result).abs().max().item()
    assert diff < 0.01, f"tile_transpose max diff {diff}"
