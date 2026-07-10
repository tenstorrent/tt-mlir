# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kernels.prefill.rope import KERNEL_BENCHES, build_rope_tables
from runner import TensorSpec, run_bench
from utils import assert_pcc


@pytest.mark.parametrize(
    "seq_len,head_dim,grid_shape,block_shape",
    [
        (64, 64, (1, 1), [2, 2]),
        (64, 128, (1, 2), [2, 2]),
    ],
)
def test_rope_matches_torch(seq_len, head_dim, grid_shape, block_shape):
    """Test rope at various workload shapes and execution configs."""

    def _cos_dist(shape, td, gen):
        cos, _ = build_rope_tables(shape[0], shape[1], start_pos=7, dtype=td)
        return cos

    def _sin_dist(shape, td, gen):
        _, sin_signed = build_rope_tables(shape[0], shape[1], start_pos=7, dtype=td)
        return sin_signed

    tensors = [
        TensorSpec(
            shape=(seq_len, head_dim),
            block_shape=block_shape,
            dtype=torch.float32,
            dist="uniform(-1,1)",
        ),
        TensorSpec(
            shape=(seq_len, head_dim),
            block_shape=block_shape,
            dtype=torch.float32,
            dist=_cos_dist,
        ),
        TensorSpec(
            shape=(seq_len, head_dim),
            block_shape=block_shape,
            dtype=torch.float32,
            dist=_sin_dist,
        ),
    ]
    actual, expected = run_bench(
        KERNEL_BENCHES["rope"], tensors=tensors, grid_shape=grid_shape
    )
    assert_pcc(expected, actual, threshold=0.99)
    assert torch.allclose(expected, actual, atol=0.05, rtol=0.05)


def test_build_rope_tables_start_pos_and_signed_sin():
    """Test rope table generation with start_pos."""
    cos, sin_signed = build_rope_tables(
        seq_len=2, head_dim=4, start_pos=3, theta=10000.0, dtype=torch.float32
    )

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 4, 2).float() / 4))
    angles = torch.outer(torch.arange(3, 5, dtype=torch.float32), inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)

    assert torch.allclose(cos, torch.cat([cos_half, cos_half], dim=-1))
    assert torch.allclose(sin_signed, torch.cat([-sin_half, sin_half], dim=-1))
