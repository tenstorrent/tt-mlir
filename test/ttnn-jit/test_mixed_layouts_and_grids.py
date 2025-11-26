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
    "shape, grid1, grid2",
    [
        # square
        # ((64, 64), (0, 0), (1, 1)),
        # ((96, 192), (0, 0), (2, 2)),
        # ((128, 128), (1, 1), (0, 0)),
        # ((192, 192), (2, 2), (0, 0)),
        ((2, 64, 64), (0, 0), (1, 1)),
        # non square
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op", [add])
def test_l1_varying_grids(device, shape, grid1, grid2, dtype, op):

    input_sharded_tensor1 = create_sharded_tile_tensor(
        device, shape, max_grid=grid1, dtype=dtype
    )
    input_sharded_tensor2 = create_sharded_tile_tensor(
        device, shape, max_grid=grid2, dtype=dtype
    )

    # core_range_1 = ttnn.CoreRangeSet({
    #     ttnn.CoreRange(
    #         ttnn.CoreCoord(0, 0),
    #         ttnn.CoreCoord(grid1[0], grid1[1]),
    #     )
    # })

    # core_range_2 = ttnn.CoreRangeSet({
    #     ttnn.CoreRange(
    #         ttnn.CoreCoord(0, 0),
    #         ttnn.CoreCoord(grid2[0], grid2[1]),
    #     )
    # })

    # tensor_spec1 = ttnn.TensorSpec(
    #     shape=(h, w),
    #     dtype=ttnn.bfloat16,
    #     layout=ttnn.TILE_LAYOUT,
    #     buffer_type=ttnn.BufferType.L1
    # ).block_sharded(core_range_1)

    # tensor_spec2 = ttnn.TensorSpec(
    #     shape=(h, w),
    #     dtype=ttnn.bfloat16,
    #     layout=ttnn.TILE_LAYOUT,
    #     buffer_type=ttnn.BufferType.L1
    # ).block_sharded(core_range_2)

    # torch_tensor1 = torch.randn((h, w)) * 100
    # torch_tensor2 = torch.randn((h, w)) * 100

    # input_sharded_tensor1 = ttnn.from_torch(
    #     torch_tensor1,
    #     spec=tensor_spec1,
    #     device=device,
    # )

    # input_sharded_tensor2 = ttnn.from_torch(
    #     torch_tensor2,
    #     spec=tensor_spec2,
    #     device=device,
    # )

    golden_op = _get_ttnn_op(op)

    op_jit = ttnn_jit.jit(debug=True)(op)
    output_tensor = op_jit(input_sharded_tensor2, input_sharded_tensor1)
    print("after jit")
    golden_tensor = (golden_op or op)(input_sharded_tensor2, input_sharded_tensor1)

    assert all_close_check(output_tensor, golden_tensor)
