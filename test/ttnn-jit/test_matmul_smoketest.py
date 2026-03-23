# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from op_definitions import matmul
from ttnn_jit._src.utils import get_maximal_block_sharding_grid
from utils import (
    create_sharded_tile_tensor,
    create_dram_tensor,
    get_core_grid_from_device,
)

# Tests with matmul in the middle / end of an op chain are failing in D2MToTTNN.
# Tests with matmul as the first op are passing. Issue #7419.
def matmul_composite(input0, input1):
    a = ttnn.abs(input0)
    b = ttnn.matmul(a, input1)
    c = ttnn.abs(b)
    return c


TILE_SIZE = 32

# None is testing the 1D matmul, single tile case
MATMUL_SHAPES = [
    ((64, 64, 64)),
    ((64, 128, 128)),
    ((64, 128, 256)),
    ((128, 128, 128)),
    ((128, 128, 256)),
    ((128, 256, 256)),
    ((256, 256, 256)),
    ((256, None, 256)),
    ((None, 256, None)),
]

INPUT_LAYOUTS = [
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.INTERLEAVED),
    (ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.INTERLEAVED),
]


@pytest.mark.parametrize(
    "shapes",
    MATMUL_SHAPES,
    ids=[f"{shape}" for shape in MATMUL_SHAPES],
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
@pytest.mark.parametrize(
    "input_layouts",
    INPUT_LAYOUTS,
    ids=[f"{str(layout)}" for layout in INPUT_LAYOUTS],
)
@pytest.mark.parametrize(
    "op",
    [
        matmul,
    ],
    ids=["matmul"],
)
def test_matmul_smoketest(device, shapes, input_layouts, dtype, ttnn_dtype, op):
    # Skip large matmuls for float32
    if dtype == torch.float32 and shapes in [(256, 256, 256), (128, 256, 256)]:
        pytest.skip("Skipping large matmul for float32")

    if ttnn_dtype == ttnn.DataType.BFLOAT8_B and shapes == (None, 256, None):
        pytest.skip(
            "Skipping test for shape (None, 256, None) with dtype bfp8, pcc error. Issue #7418."
        )

    # Always square grid.
    core_grid = get_core_grid_from_device(device)
    grid_dim = min(core_grid[0] + 1, core_grid[1] + 1)
    core_grid = (grid_dim - 1, grid_dim - 1)
    m = TILE_SIZE if shapes[0] is None else shapes[0] * grid_dim
    k = TILE_SIZE if shapes[1] is None else shapes[1] * grid_dim
    n = TILE_SIZE if shapes[2] is None else shapes[2] * grid_dim

    # input is (m, k, n)
    shapes = [(m, k), (k, n)]
    input_tensors = []
    for shape, layout in zip(shapes, input_layouts):
        if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            grid = get_maximal_block_sharding_grid(shape, core_grid)
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
    )(op)

    output = compiled_op(*input_tensors)
    assert output.memory_config().is_sharded(), "Matmul output must be sharded"

    # Send tensor to DRAM to avoid having to set the matmul program config in the golden path
    input_tensors = [
        ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        for tensor in input_tensors
    ]
    golden_output = op(*input_tensors)
    pcc = ttnn.pearson_correlation_coefficient(
        golden_output.cpu().to_torch(), output.cpu().to_torch()
    )
    print("pcc: ", pcc)
    assert pcc > 0.99, f"PCC: {pcc} is less than 0.99"


def test_matmul_f32_fallback(device):
    """f32 (256,256,256) BLOCK_SHARDED+INTERLEAVED fails JIT pipeline, fallback works."""
    core_grid = get_core_grid_from_device(device)
    grid_dim = min(core_grid[0] + 1, core_grid[1] + 1)
    core_grid = (grid_dim - 1, grid_dim - 1)

    m = 256 * grid_dim
    k = 256 * grid_dim
    n = 256 * grid_dim

    grid0 = get_maximal_block_sharding_grid((m, k), core_grid)
    input0 = create_sharded_tile_tensor(
        device, (m, k), grid0, torch.float32, shard_strategy=ttnn.ShardStrategy.BLOCK
    )
    input1 = create_dram_tensor(device, (k, n), torch.float32)

    compiled_op = ttnn_jit.jit(debug=True, compile_only=False, fallback=True)(matmul)
    output = compiled_op(input0, input1)

    assert compiled_op.fallback_used, "Expected fallback for this configuration"

    input0_dram = ttnn.to_memory_config(input0, ttnn.DRAM_MEMORY_CONFIG)
    input1_dram = ttnn.to_memory_config(input1, ttnn.DRAM_MEMORY_CONFIG)
    golden = matmul(input0_dram, input1_dram)
    pcc = ttnn.pearson_correlation_coefficient(
        golden.cpu().to_torch(), output.cpu().to_torch()
    )
    print("pcc: ", pcc)
    assert pcc > 0.99, f"PCC: {pcc} is less than 0.99"
