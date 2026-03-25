# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Parametrized matmul perf benchmarks comparing JIT vs TTNN across
# various shapes, memory configurations, data types, and math fidelities.

import ttnn
import ttnn_jit
import torch

import pytest

from op_definitions import matmul
from utils import create_sharded_tile_tensor, create_dram_tensor

MATMUL_SHAPES = [
    (512, 512, 512),
    (512, 1024, 1024),
    (512, 1024, 2048),
    (1024, 1024, 512),
    (1024, 1024, 1024),
    (1024, 1024, 2048),
    (1024, 2048, 2048),
    (2048, 1024, 512),
    (2048, 1024, 1024),
    (2048, 2048, 1024),
    (2048, 2048, 2048),
]

GRID_8x8 = (7, 7)

# (memory_config_id, input_a_mem, input_b_mem)
# TTNN matmul requires input_b to have INTERLEAVED memory layout.
MEMORY_CONFIGS = [
    ("l1_dram", "l1_block_sharded", "dram_interleaved"),
    ("dram_interleaved", "dram_interleaved", "dram_interleaved"),
]


def _create_tensor(device, shape, dtype, ttnn_dtype, mem_type):
    if mem_type == "l1_block_sharded":
        return create_sharded_tile_tensor(
            device,
            shape,
            GRID_8x8,
            dtype,
            shard_strategy=ttnn.ShardStrategy.BLOCK,
            ttnn_dtype=ttnn_dtype,
        )
    return create_dram_tensor(device, shape, dtype, ttnn_dtype=ttnn_dtype)


@pytest.mark.parametrize(
    "m, k, n",
    MATMUL_SHAPES,
    ids=[f"{m}x{k}x{n}" for m, k, n in MATMUL_SHAPES],
)
@pytest.mark.parametrize(
    "op",
    [matmul],
    ids=["matmul"],
)
@pytest.mark.parametrize(
    "memory_config_id, input_a_mem, input_b_mem",
    MEMORY_CONFIGS,
    ids=[cfg[0] for cfg in MEMORY_CONFIGS],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype, math_fidelity",
    [
        (torch.bfloat16, ttnn.DataType.BFLOAT16, ttnn.MathFidelity.HiFi4),
        (torch.bfloat16, ttnn.DataType.BFLOAT8_B, ttnn.MathFidelity.HiFi2),
    ],
    ids=["bf16_hifi4", "bfp8_hifi2"],
)
@pytest.mark.parametrize(
    "jit_enabled",
    [True, False],
)
def test_matmul_perf(
    m,
    k,
    n,
    op,
    memory_config_id,
    input_a_mem,
    input_b_mem,
    dtype,
    ttnn_dtype,
    math_fidelity,
    jit_enabled,
    perf_device,
):
    device = perf_device
    shape_a = (m, k)
    shape_b = (k, n)

    input_a = _create_tensor(device, shape_a, dtype, ttnn_dtype, input_a_mem)
    input_b = _create_tensor(device, shape_b, dtype, ttnn_dtype, input_b_mem)

    if jit_enabled:
        function_to_test = ttnn_jit.jit(
            enable_cache=True,
            math_fidelity=math_fidelity,
        )(op)
    else:
        function_to_test = op

    output = function_to_test(input_a, input_b)
    assert output is not None
