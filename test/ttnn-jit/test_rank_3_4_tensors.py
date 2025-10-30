# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    _get_ttnn_op,
    all_close_check,
    memory_configs_equal,
)

import ast
import inspect

from ttmlir.ir import *
from ttmlir.dialects import (
    ttir,
    func,
    ttnn,
    tensor,
    ttcore,
)

def abs(input_tensor):
    return ttnn.abs(input_tensor)

def create_rank_n_sharded_tile_tensor(device, shape, max_grid, dtype=torch.float32, int_max=0):
    torch.manual_seed(0)
    if not (dtype.is_floating_point or dtype.is_complex):
        # recreate spatial coverage of fp [0,1] in randn and give some overflow headroom
        high_val = int_max if int_max else torch.iinfo(dtype).max // 2
        torch_tensor = torch.randint(high_val, shape, dtype=dtype)
    else:
        if int_max:
            print("Warning: int_max provided for floating point tensor, ignoring.")
        torch_tensor = torch.randn(shape, dtype=dtype)

    start_coord = ttnn.CoreCoord(0, 0)
    end_coord = ttnn.CoreCoord(max_grid[0], max_grid[1])
    core_range = ttnn.CoreRange(start_coord, end_coord)
    core_range_set = ttnn.CoreRangeSet([core_range])

    # TTNN grids are (Width, Height), while tensor shapes are (Height, Width).
    shard_shape_x = h if max_grid[1] == 0 else h // (max_grid[1] + 1)
    shard_shape_y = w if max_grid[0] == 0 else w // (max_grid[0] + 1)

    shard_spec = ttnn.ShardSpec(
        grid=core_range_set,
        shard_shape=[shard_shape_x, shard_shape_y],
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    return ttnn.from_torch(
        torch_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

#rank 3 tensor test
@pytest.mark.parametrize("rank_1, rank_2, rank_3", [(4, 5, 6), (2, 3, 4)])
@pytest.mark.parametrize("max_grid", (7, 7))
@pytest.mark.parametrize("op", [abs])
def test_rank_3(device, rank_1, rank_2, rank_3, max_grid, op):
    input_tensor = create_rank_n_sharded_tile_tensor(
        device,
        (rank_1, rank_2, rank_3),
        max_grid,
        dtype=torch.float32,
    )

    # JIT compile the operation
    jit_op = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(op)
    output_tensor = op_jit(input_tensor)

    golden_tensor = op(input_tensor)

    assert memory_configs_equal(output_tensor.memory_config(), golden_tensor.memory_config())
    assert all_close_check(output_tensor, golden_tensor)

