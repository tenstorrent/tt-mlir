# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import math
import pytest

from utils import (
    create_sharded_tile_tensor,
    create_dram_tensor,
    get_core_grid_from_device,
)


def matmul_composite(input0, input1):
    a = ttnn.abs(input0)
    b = ttnn.sin(input1)
    c = ttnn.matmul(a, b)
    d = ttnn.abs(c)
    return d


MATMUL_SHAPES = [
    ((64, 64, 64)),
    ((64, 128, 128)),
    ((64, 128, 256)),
    ((128, 128, 128)),
    ((128, 128, 256)),
    ((128, 256, 256)),
    ((256, 256, 256)),
    ((256, 4, 256)),
    ((4, 256, 4)),
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
def test_matmul_composite(device, shapes, input_layouts, dtype, ttnn_dtype):
    # Skip large matmuls for float32
    if dtype == torch.float32 and shapes in [(256, 256, 256), (128, 256, 256)]:
        pytest.skip("Skipping large matmul for float32")

    # Always square grid.
    core_grid = get_core_grid_from_device(device)
    grid_size = math.lcm(core_grid[0] + 1, core_grid[1] + 1)
    m = shapes[0] * grid_size
    k = shapes[1] * grid_size
    n = shapes[2] * grid_size

    # input is (m, k, n)
    shapes = [(m, k), (k, n)]
    input_tensors = []
    for shape, layout in zip(shapes, input_layouts):
        if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            input_tensors.append(
                create_sharded_tile_tensor(
                    device,
                    shape,
                    core_grid,
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
    )(matmul_composite)

    output = compiled_op(*input_tensors)
    assert output.memory_config().is_sharded(), "Matmul output must be sharded"

    # Send tensor to DRAM to avoid having to set the matmul program config in the golden path
    input_tensors = [
        ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        for tensor in input_tensors
    ]
    golden_output = matmul_composite(*input_tensors)
    pcc = ttnn.pearson_correlation_coefficient(
        golden_output.cpu().to_torch(), output.cpu().to_torch()
    )
    print("pcc: ", pcc)
    assert pcc > 0.99, f"PCC: {pcc} is less than 0.99"
