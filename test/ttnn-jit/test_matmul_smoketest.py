# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch
import itertools
import os

import pytest

from ttnn_jit._src.utils import get_maximal_block_sharding_grid
from utils import (
    create_sharded_tile_tensor,
    create_dram_tensor,
    get_core_grid_from_device,
)


def matmul_composite(input0, input1):
    # a = ttnn.abs(input0)
    # b = ttnn.sin(input1)
    # c = ttnn.matmul(a, b)
    # d = ttnn.abs(c)
    # return d
    return ttnn.matmul(input0, input1)


MATMUL_SHAPES = [
    ((512, 512, 512)),
    ((512, 1024, 1024)),
    ((512, 1024, 2048)),
    ((1024, 1024, 1024)),
    ((1024, 1024, 2048)),
    ((1024, 2048, 2048)),
    ((2048, 2048, 2048)),
    # ((2048, 32, 2048)),
    # ((32, 2048, 32)),
]

# for 11x10 grid, same # elements per core as MATMUL_SHAPES
BLACKHOLE_MATMUL_SHAPES = [
    ((640, 704, 704)),          # 64 64 64
    ((640, 1408, 1408)),        # 64 128 128
    ((640, 1408, 2816)),        # 64 128 256
    ((1280, 1408, 1408)),       # 128 128 128
    ((1280, 1408, 2816)),       # 128 128 256
    ((1280, 2816, 2816)),       # 128 256 256
    ((2560, 2816, 2816)),       # 256 256 256
]

# for 10x10 grid, same # elements per core as MATMUL_SHAPES
BLACKHOLE_MATMUL_SHAPES_10x10 = [
    ((640, 640, 640)),          # 64 64 64
    ((640, 1280, 1280)),        # 64 128 128
    ((640, 1280, 2560)),        # 64 128 256
    ((1280, 1280, 1280)),       # 128 128 128
    ((1280, 1280, 2560)),       # 128 128 256
    ((1280, 2560, 2560)),       # 128 256 256
    ((2560, 2560, 2560)),       # 256 256 256
]

INPUT_LAYOUTS = [
    # (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.INTERLEAVED),
    # (ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # (ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.INTERLEAVED),
]


@pytest.mark.parametrize(
    "shapes",
    BLACKHOLE_MATMUL_SHAPES_10x10,
    ids=[f"{shape}" for shape in BLACKHOLE_MATMUL_SHAPES_10x10],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        # (torch.float32, None),
        (torch.bfloat16, None),
        # (torch.bfloat16, ttnn.DataType.BFLOAT8_B),
    ],
    # ids=["f32", "bf16", "bfp8"],
    ids=["bf16"],
)
@pytest.mark.parametrize(
    "input_layouts",
    INPUT_LAYOUTS,
    ids=[f"{str(layout)}" for layout in INPUT_LAYOUTS],
)
def test_matmul_composite(device, shapes, input_layouts, dtype, ttnn_dtype):
    # Skip large matmuls for float32
    if dtype == torch.float32 and shapes == (2048, 2048, 2048):
        pytest.skip("Skipping large matmul for float32")
    # input is (m, k, n)
    shapes = [(shapes[0], shapes[1]), (shapes[1], shapes[2])]
    input_tensors = []
    for shape, layout in zip(shapes, input_layouts):
        if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            grid = get_maximal_block_sharding_grid(
                shape, get_core_grid_from_device(device)
            )
            print(f"grid: {grid}")
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
        enable_l1_acc=True,
        use_tile_matmul=False,
        # math_fidelity=ttnn.MathFidelity.HiFi2
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
