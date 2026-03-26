# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    run_op_test,
)


REDUCTION_SHAPES = [
    # (shape, max_grid, dim)
    ((32, 32), (0, 0), 1),
    ((128, 128), (0, 0), 0),
    ((256, 256), (0, 0), 1),
    ((128, 64), (1, 0), 0),
    ((512, 1024), (7, 0), 0),
    ((64, 32), (0, 1), 1),
    ((1024, 1024), (0, 7), 1),
]

REDUCTION_OPS = [
    ttnn.max,
    ttnn.sum,
    ttnn.mean,
    ttnn.min,
]

# L1 block-sharded min has PCC issues due to OOB padding values in the neg→max→neg decomposition.
# Also the DRAM test fails with operand #0 does not dominate this use.
# TODO(sgholami): Investigate L1 block-sharded min PCC failures. Issue: #7617
SKIP_MIN_TESTS = [
    (ttnn.BufferType.L1, (128, 128), (0, 0)),
    (ttnn.BufferType.L1, (256, 256), (0, 0)),
    (ttnn.BufferType.L1, (128, 64), (1, 0)),
    (ttnn.BufferType.DRAM, (32, 32), (0, 0)),
]


@pytest.mark.parametrize("shape, max_grid, dim", REDUCTION_SHAPES)
@pytest.mark.parametrize("op", REDUCTION_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_reductions(device, shape, max_grid, dim, op, dtype, buffer_type):
    """Test reduction operations (max, sum) with L1 or DRAM config."""

    def reduction_func(input_tensor):
        return op(input_tensor, dim=dim, keepdim=True)

    if op == ttnn.min and (buffer_type, shape, max_grid) in SKIP_MIN_TESTS:
        pytest.skip(reason="L1 block-sharded min has PCC issues (see #7617)")

    shard_strategy = (
        ttnn.ShardStrategy.BLOCK if buffer_type == ttnn.BufferType.L1 else None
    )

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        reduction_func,
        num_inputs=1,
        buffer_type=buffer_type,
        shard_strategy=shard_strategy,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
