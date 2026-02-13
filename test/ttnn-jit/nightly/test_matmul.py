# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch
import itertools

import pytest

from ttnn_jit._src.utils import get_maximal_block_sharding_grid
from utils import (
    create_sharded_tile_tensor,
    create_dram_tensor,
    get_core_grid_from_device,
)

from op_definitions import (
    matmul,
)


# Test full grid and high aspect ratio grids
MATMUL_SHAPE_GRIDS_SINGLE_OR_FULL = [
    (
        (m * 32 * (grid_m + 1), k * 32 * (grid_k + 1), n * 32 * (grid_n + 1)),
        (grid_m, grid_k, grid_n),
    )
    for m, k, n, grid_m, grid_k, grid_n in itertools.product(
        [4], [4], [4], [0, 7], [0, 7], [0, 7]
    )
]


@pytest.mark.parametrize(
    "shape_grids",
    MATMUL_SHAPE_GRIDS_SINGLE_OR_FULL,
    ids=[f"{shape}-{grid}" for shape, grid in MATMUL_SHAPE_GRIDS_SINGLE_OR_FULL],
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
def test_matmul_with_dtypes(device, shape_grids, dtype, ttnn_dtype):
    shapes, grids = shape_grids
    # shape is (m, k, n)
    shape0 = [shapes[0], shapes[1]]
    shape1 = [shapes[1], shapes[2]]
    # grid is (grid_m, grid_k, grid_n)
    grid0 = [grids[1], grids[0]]
    grid1 = [grids[2], grids[1]]

    compiled_op = ttnn_jit.jit(
        debug=True,
        compile_only=False,
    )(matmul)
    input0_tensor = create_sharded_tile_tensor(
        device,
        shape0,
        grid0,
        dtype,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
        ttnn_dtype=ttnn_dtype,
    )
    input1_tensor = create_sharded_tile_tensor(
        device,
        shape1,
        grid1,
        dtype,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
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


MATMUL_SHAPES = [
    (m * 32, k * 32, n * 32)
    for m, k, n in itertools.product(range(1, 9), range(1, 9), range(1, 9))
]

INPUT_LAYOUTS = [
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.INTERLEAVED),
    (ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.INTERLEAVED),
]


@pytest.mark.parametrize(
    "shapes",
    MATMUL_SHAPES,
    ids=[f"{str(shape)}" for shape in MATMUL_SHAPES],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.bfloat16, None),
    ],
    ids=["bf16"],
)
@pytest.mark.parametrize(
    "input_layouts",
    INPUT_LAYOUTS,
    ids=[f"{str(layout)}" for layout in INPUT_LAYOUTS],
)
def test_matmul_with_grids(device, shapes, dtype, ttnn_dtype, input_layouts):
    shapes = [(shapes[0], shapes[1]), (shapes[1], shapes[2])]
    input_tensors = []
    for shape, layout in zip(shapes, input_layouts):
        if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            grid = get_maximal_block_sharding_grid(
                shape, get_core_grid_from_device(device)
            )
            input_tensors.append(
                create_sharded_tile_tensor(
                    device,
                    shape,
                    grid,
                    dtype,
                    shard_strategy=ttnn.ShardStrategy.BLOCK,
                    ttnn_dtype=ttnn_dtype,
                )
            )
        else:
            input_tensors.append(
                create_dram_tensor(device, shape, dtype, ttnn_dtype=ttnn_dtype)
            )

    compiled_op = ttnn_jit.jit(
        debug=True,
        compile_only=False,
    )(matmul)
    output = compiled_op(*input_tensors)
    assert output.memory_config().is_sharded(), "Matmul output must be sharded"

    # Send tensor to DRAM to avoid having to set the matmul program config in the golden path
    input_tensors = [
        ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        for tensor in input_tensors
    ]
    golden_output = ttnn.matmul(*input_tensors)
    pcc = ttnn.pearson_correlation_coefficient(
        golden_output.cpu().to_torch(), output.cpu().to_torch()
    )
    print("pcc: ", pcc)
    assert pcc > 0.99, f"PCC: {pcc} is less than 0.99"
