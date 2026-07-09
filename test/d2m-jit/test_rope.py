# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import d2m_jit as d2m
from kernels.prefill.rope import apply_rope, build_rope_tables
from utils import assert_pcc


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _unsigned_sin(sin_signed):
    half = sin_signed.shape[-1] // 2
    return torch.cat([-sin_signed[..., :half], sin_signed[..., half:]], dim=-1)


@pytest.mark.parametrize(
    "seq_len,head_dim,block_shape,grid_shape,grid",
    [
        (64, 64, [2, 2], [1, 1], (1, 1)),
        (64, 128, [2, 2], [1, 2], (1, 2)),
    ],
)
def test_rope_matches_torch(seq_len, head_dim, block_shape, grid_shape, grid):
    x = torch.randn(seq_len, head_dim, dtype=torch.float32)
    cos, sin_signed = build_rope_tables(
        seq_len, head_dim, start_pos=7, dtype=torch.float32
    )
    sin = _unsigned_sin(sin_signed)

    layout = d2m.Layout(
        shape=x.shape,
        dtype=x.dtype,
        block_shape=block_shape,
        grid_shape=grid_shape,
    )
    x_lt = d2m.to_layout(x, layout)
    cos_lt = d2m.to_layout(cos, layout)
    sin_signed_lt = d2m.to_layout(sin_signed, layout)

    out = apply_rope(
        x_lt,
        cos_lt,
        sin_signed_lt,
        layout,
        grid,
        m_blocks=1,
        n_blocks=1,
    ).to_host()

    golden = x * cos + rotate_half(x) * sin
    assert_pcc(golden, out)
    assert torch.allclose(golden, out, atol=0.05, rtol=0.05)


def test_build_rope_tables_start_pos_and_signed_sin():
    cos, sin_signed = build_rope_tables(
        seq_len=2, head_dim=4, start_pos=3, theta=10000.0, dtype=torch.float32
    )

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 4, 2).float() / 4))
    angles = torch.outer(torch.arange(3, 5, dtype=torch.float32), inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)

    assert torch.allclose(cos, torch.cat([cos_half, cos_half], dim=-1))
    assert torch.allclose(sin_signed, torch.cat([-sin_half, sin_half], dim=-1))
