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
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)

def add(input_tensor1, input_tensor2):
    return ttnn.add(input_tensor1, input_tensor2)

@pytest.mark.parametrize(
    "h , w, max_grid",
    [
        (32, 32, (0, 0)),
        (32, 64, (0, 0)),
        (64, 64, (0, 0)),
        (64, 128, (0, 0)),
    ]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op", [add])
def test_mixed_memory_layouts(device, h, w, max_grid, dtype, op):

    input_dram_tensor = create_dram_tensor(device, h, w, dtype)
    input_sharded_tensor = create_sharded_tile_tensor(device, h, w, max_grid=max_grid, dtype=dtype)

    golden_op = _get_ttnn_op(op)
    
    op_jit = ttnn_jit.jit(
        debug=True,
        max_grid=max_grid
    )(op)
    output_tensor = op_jit(input_dram_tensor, input_sharded_tensor)
    golden_tensor = (golden_op or op)(input_dram_tensor, input_sharded_tensor)

    assert all_close_check(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "h , w, grid1, grid2",
    [
        (64, 64, (0, 0), (1, 1)),
        (96, 192, (0, 0), (2, 2)),
        (128, 128, (1, 1), (0, 0)),
        (192, 192, (2, 2), (0, 0)),
    ]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op", [add])
def test_l1_varying_grids(device, h, w, grid1, grid2, dtype, op):

    input_sharded_tensor1 = create_sharded_tile_tensor(device, h, w, max_grid=grid1, dtype=dtype)
    input_sharded_tensor2 = create_sharded_tile_tensor(device, h, w, max_grid=grid2, dtype=dtype)

    golden_op = _get_ttnn_op(op)
    
    op_jit = ttnn_jit.jit(
        debug=True,
        max_grid=max(grid1, grid2)
    )(op)
    output_tensor = op_jit(input_sharded_tensor1, input_sharded_tensor2)
    golden_tensor = (golden_op or op)(input_sharded_tensor1, input_sharded_tensor2)

    assert memory_configs_equal(
        output_tensor.memory_config(), golden_tensor.memory_config()
    )
    assert all_close_check(output_tensor, golden_tensor)