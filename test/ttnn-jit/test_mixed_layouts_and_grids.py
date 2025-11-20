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
        # square
        (32, 32, (0, 0)),
        (64, 64, (0, 0)),
        (256, 256, (7, 7)),
        (512, 512, (7, 7)),
        (128, 128, (3, 3)),
        (256, 256, (3, 3)),
        # non-square
        (32, 64, (0, 0)),
        (64, 128, (0, 0)),
        (256, 512, (3, 3)),
        (512, 1024, (3, 3)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("op", [add])
def test_mixed_memory_layouts(device, h, w, max_grid, dtype, op):

    input_dram_tensor = create_dram_tensor(device, h, w, dtype)
    input_sharded_tensor = create_sharded_tile_tensor(
        device, h, w, max_grid=max_grid, dtype=dtype
    )

    golden_op = _get_ttnn_op(op)

    op_jit = ttnn_jit.jit(debug=True, max_grid=(0, 0))(op)
    output_tensor = op_jit(input_sharded_tensor, input_dram_tensor)
    golden_tensor = (golden_op or op)(input_dram_tensor, input_sharded_tensor)

    assert all_close_check(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "h , w, grid1, grid2",
    [
        # square
        (64, 64, (0, 0), (1, 1)),
        (96, 192, (0, 0), (2, 2)),
        (128, 128, (1, 1), (0, 0)),
        (192, 192, (2, 2), (0, 0)),
        # non square
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op", [add])
def test_l1_varying_grids(device, h, w, grid1, grid2, dtype, op):

    input_sharded_tensor1 = create_sharded_tile_tensor(
        device, h, w, max_grid=grid1, dtype=dtype
    )
    input_sharded_tensor2 = create_sharded_tile_tensor(
        device, h, w, max_grid=grid2, dtype=dtype
    )

    golden_op = _get_ttnn_op(op)

    grid1_size = (grid1[0] + 1) * (grid1[1] + 1)
    grid2_size = (grid2[0] + 1) * (grid2[1] + 1)
    max_grid = grid1 if grid1_size >= grid2_size else grid2

    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(op)
    output_tensor = op_jit(input_sharded_tensor1, input_sharded_tensor2)
    print("after jit")
    golden_tensor = (golden_op or op)(input_sharded_tensor1, input_sharded_tensor2)

    assert all_close_check(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "h , w, max_grid",
    [
        # square
        (32, 32, (0, 0)),
        (64, 64, (0, 0)),
        (256, 256, (7, 7)),
        (512, 512, (7, 7)),
        (128, 128, (3, 3)),
        (256, 256, (3, 3)),
        # non-square
        # (32, 64, (0, 0)),
        # (64, 128, (0, 0)),
        # ((0, 3))
        # (())
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("op", [add])
def test_ttnn_mixed_memory_layouts(device, h, w, max_grid, dtype, op):

    input_dram_tensor = create_dram_tensor(device, h, w, dtype)
    input_sharded_tensor = create_sharded_tile_tensor(
        device, h, w, max_grid=max_grid, dtype=dtype
    )

    golden_op = _get_ttnn_op(op)

    # op_jit = ttnn_jit.jit(
    #     debug=True,
    #     max_grid=(0, 0)
    # )(op)
    # output_tensor = op_jit(input_dram_tensor, input_sharded_tensor)
    golden_tensor = (golden_op or op)(input_dram_tensor, input_sharded_tensor)
    print(golden_tensor)


@pytest.mark.parametrize(
    "h , w, grid1, grid2",
    [
        # square
        (64, 64, (0, 0), (1, 1)),
        (96, 192, (0, 0), (2, 2)),
        (128, 128, (1, 1), (0, 0)),
        (192, 192, (2, 2), (0, 0)),
        # non square
    ],
)
def test_ttnn_l1_varying_grids(device, h, w, grid1, grid2):

    input_sharded_tensor1 = create_sharded_tile_tensor(
        device, h, w, max_grid=grid1, dtype=torch.bfloat16
    )
    input_sharded_tensor2 = create_sharded_tile_tensor(
        device, h, w, max_grid=grid2, dtype=torch.bfloat16
    )

    print("input tensor 1:", input_sharded_tensor1)
    print("input tensor 2:", input_sharded_tensor2)

    golden_tensor = input_sharded_tensor1 + input_sharded_tensor2
    print("golden tensor:", golden_tensor)

    print(golden_tensor.cpu().to_torch())
