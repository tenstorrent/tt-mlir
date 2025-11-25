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

# WIthout support for JIT'ed bin ops with inputs with differing sharding grids,
# much of the testing here should be expected to fail. The trivial case is when
# max_grid = (0, 0) for both inputs ->
HEIGHT_WIDTH_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    # ((2048, 2048), (7, 7)),  #fails. error: 'd2m.generic' op grid shape mismatch between operand[0] grid_shape=[64, 1] and operand[1] grid_shape=[1, 64] at affine dim d0
    ((2, 32, 64), (0, 0)),
    ((2, 64, 128), (0, 0)),
    # ((32, 64, 2048), (7, 7)), #fails. error: 'd2m.generic' op grid shape mismatch between operand[0] grid_shape=[64, 1] and operand[1] grid_shape=[1, 64] at affine dim d0
    ((128, 128), (0, 0)),
    (
        (1, 4, 16, 128),
        (0, 0),
    ),  # fails. error: Number of shards along height 2 must not exceed number of cores 1
]

HEIGHT_BLOCK_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((256, 64), (7, 0)),
    ((256, 64), (0, 7)),
    ((2048, 128), (7, 7)),
    ((384, 32), (1, 5)),
    ((2, 192, 32), (1, 5)),
    ((2, 2, 96, 32), (1, 5)),
    ((2, 2, 512, 32), (7, 7)),
]

WIDTH_BLOCK_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((64, 256), (7, 0)),
    ((64, 256), (0, 7)),
    ((128, 2048), (7, 7)),
    ((32, 384), (1, 5)),
    ((2, 32, 384), (1, 5)),
    ((2, 2, 32, 384), (1, 5)),
    ((2, 1, 32, 2048), (7, 7)),
]


def add(input_tensor_a, input_tensor_b):
    return ttnn.add(input_tensor_a, input_tensor_b)


@pytest.mark.parametrize(
    "shape, max_grid",
    HEIGHT_WIDTH_SHARDED_SHAPE_GRIDS,
    ids=[
        f"shape={shape}_grid={max_grid}"
        for shape, max_grid in HEIGHT_WIDTH_SHARDED_SHAPE_GRIDS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("op", [add])
@pytest.mark.parametrize("graph_capture", [True, False])
def test_height_width_mixed_legacy_sharding_types(
    device, shape, max_grid, dtype, op, graph_capture
):
    # Create input tensors
    input_tensor_a = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    input_tensor_b = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )

    op_jit = ttnn_jit.jit(
        debug=True,
        max_grid=max_grid,
        graph_capture=graph_capture,
    )(op)
    output_tensor = op_jit(input_tensor_a, input_tensor_b)

    golden_output = op(input_tensor_a, input_tensor_b)

    assert all_close_check(output_tensor, golden_output)


@pytest.mark.parametrize(
    "shape, max_grid",
    HEIGHT_BLOCK_SHARDED_SHAPE_GRIDS,
    ids=[
        f"shape={shape}_grid={max_grid}"
        for shape, max_grid in HEIGHT_BLOCK_SHARDED_SHAPE_GRIDS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("op", [add])
@pytest.mark.parametrize("graph_capture", [True, False])
def test_height_block_mixed_legacy_sharding_types(
    device, shape, max_grid, dtype, op, graph_capture
):
    # Create input tensors
    input_tensor_a = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    input_tensor_b = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )

    op_jit = ttnn_jit.jit(
        debug=True,
        max_grid=max_grid,
        graph_capture=graph_capture,
    )(op)
    output_tensor = op_jit(input_tensor_a, input_tensor_b)

    golden_output = op(input_tensor_a, input_tensor_b)

    assert all_close_check(output_tensor, golden_output)


@pytest.mark.parametrize(
    "shape, max_grid",
    WIDTH_BLOCK_SHARDED_SHAPE_GRIDS,
    ids=[
        f"shape={shape}_grid={max_grid}"
        for shape, max_grid in WIDTH_BLOCK_SHARDED_SHAPE_GRIDS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("op", [add])
@pytest.mark.parametrize("graph_capture", [True, False])
def test_width_block_mixed_legacy_sharding_types(
    device, shape, max_grid, dtype, op, graph_capture
):
    # Create input tensors
    input_tensor_a = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )
    input_tensor_b = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )

    op_jit = ttnn_jit.jit(
        debug=True,
        max_grid=max_grid,
        graph_capture=graph_capture,
    )(op)
    output_tensor = op_jit(input_tensor_a, input_tensor_b)

    golden_output = op(input_tensor_a, input_tensor_b)

    assert all_close_check(output_tensor, golden_output)
