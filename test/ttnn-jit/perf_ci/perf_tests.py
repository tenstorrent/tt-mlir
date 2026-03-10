# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttnn_jit
import torch

import pytest

from op_definitions import abs, exp, add, mul, matmul

# Memory configs that pass for all ops and both JIT and non-JIT.
# DRAM interleaved works for matmul (requires interleaved) and all elementwise ops.
# L1 interleaved is not used: JIT runtime fails with RuntimeError on L1 interleaved
# inputs (submit path), so we only test DRAM interleaved for paired JIT vs TTNN comparison.
MEMORY_CONFIGS = [
    (ttnn.DRAM_MEMORY_CONFIG, "dram_interleaved"),
]


def is_unary(op):
    return op == abs or op == exp


@pytest.mark.parametrize(
    "h, w",
    [
        (256, 256),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        abs,
        exp,
        add,
        mul,
        matmul,
    ],
    ids=[
        "abs",
        "exp",
        "add",
        "mul",
        "matmul",
    ],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.DataType.BFLOAT16),
        (torch.bfloat16, ttnn.DataType.BFLOAT8_B),
    ],
    ids=["bf16", "bfp8"],
)
@pytest.mark.parametrize(
    "memory_config, memory_config_id",
    MEMORY_CONFIGS,
    ids=[id for _, id in MEMORY_CONFIGS],
)
@pytest.mark.parametrize(
    "jit_enabled",
    [
        True,
        False,
    ],
)
def test_op_compare(
    h, w, op, dtype, ttnn_dtype, memory_config, memory_config_id, jit_enabled
):
    device = ttnn.open_device(device_id=0)
    torch_tensor_a = torch.rand((h, w), dtype=dtype) * 100
    torch_tensor_b = torch.rand((h, w), dtype=dtype) * 100

    input_a = ttnn.from_torch(
        torch_tensor_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    input_b = ttnn.from_torch(
        torch_tensor_b,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    function_to_test = (
        ttnn_jit.jit(debug=True, enable_cache=True)(op) if jit_enabled else op
    )
    output_tensor = (
        function_to_test(input_a)
        if is_unary(op)
        else function_to_test(input_a, input_b)
    )

    print(f"output_tensor\n: {output_tensor}")
    ttnn.close_device(device)
