# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch
import itertools

import pytest

from utils import (
    create_sharded_tile_tensor,
)


def matmul(input0, input1):
    return ttnn.matmul(input0, input1)

# Generate matmul shapes and grids. All layouts are block sharded
MATMUL_SHAPE_GRIDS = [
    (
        (m * 32 * (grid_m + 1), k * 32 * (grid_k + 1), n * 32 * (grid_n + 1)),
        (grid_m, grid_k, grid_n),
    )
    for m, k, n, grid_m, grid_k, grid_n in itertools.product(
        range(1, 2), range(1, 2), range(1, 2), range(0, 8), range(0, 8), range(0, 8)
    )
]


@pytest.mark.parametrize(
    "shape_grids",
    MATMUL_SHAPE_GRIDS,
    ids=[f"{shape}-{grid}" for shape, grid in MATMUL_SHAPE_GRIDS],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.float32, None),
        (torch.bfloat16, None),
        (torch.bfloat16, ttnn.DataType.BFLOAT8_B),
    ],
    ids=["f32", "bf16", "bfp8"],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_matmul(device, shape_grids, dtype, ttnn_dtype, graph_capture):
    shapes, grids = shape_grids
    # shape is (m, k, n)
    shape0 = [shapes[0], shapes[1]]
    shape1 = [shapes[1], shapes[2]]
    # grid is (grid_m, grid_k, grid_n)
    grid0 = [grids[1], grids[0]]
    grid1 = [grids[2], grids[1]]

    compiled_op = ttnn_jit.jit(
        debug=True,
        graph_capture=graph_capture,
        compile_only=False,
    )(matmul)
    input0_tensor = create_sharded_tile_tensor(
        device,
        shape0,
        grid0,
        dtype,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn_dtype=ttnn_dtype,
    )
    input1_tensor = create_sharded_tile_tensor(
        device,
        shape1,
        grid1,
        dtype,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn_dtype=ttnn_dtype,
    )
    output = compiled_op(input0_tensor, input1_tensor)
    # Send tensor to DRAM to avoid having to set the matmul program config in the golden path
    input0_tensor = ttnn.to_memory_config(input0_tensor, ttnn.DRAM_MEMORY_CONFIG)
    input1_tensor = ttnn.to_memory_config(input1_tensor, ttnn.DRAM_MEMORY_CONFIG)
    golden_output = ttnn.matmul(input0_tensor, input1_tensor)
    pcc = ttnn.pearson_correlation_coefficient(
        golden_output.cpu().to_torch(), output.cpu().to_torch()
    )
    print("pcc: ", pcc)
    assert pcc > 0.99, f"PCC: {pcc} is less than 0.99"