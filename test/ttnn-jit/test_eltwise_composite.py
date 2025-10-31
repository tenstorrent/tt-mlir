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
)

COMMON_SHAPE_GRID_PARAMS = [
    (32, 32, (0, 0)),
    (32, 64, (0, 0)),
    (64, 64, (0, 0)),
    (64, 128, (0, 0)),
    (128, 128, (0, 0)),
    (256, 256, (7, 7)),
    (256, 512, (7, 7)),
    (512, 512, (7, 7)),
    (512, 1024, (7, 7)),
    (1024, 1024, (7, 7)),
    (1024, 2048, (7, 7)),
]

COMMON_SHAPE_PARAMS = [(32, 32), (32, 64), (64, 64), (64, 128), (128, 128)]


def run_op_test(
    device, h, w, max_grid, dtype, op, num_inputs, buffer_type=ttnn.BufferType.L1
):
    if buffer_type == ttnn.BufferType.L1:
        inputs = [
            create_sharded_tile_tensor(device, h, w, max_grid, dtype)
            for _ in range(num_inputs)
        ]
    else:
        inputs = [create_dram_tensor(device, h, w, dtype) for _ in range(num_inputs)]
    print("inputs", inputs)
    golden_op = _get_ttnn_op(op)

    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(op)
    output_tensor = op_jit(*inputs)
    golden_tensor = (golden_op or op)(*inputs)

    assert memory_configs_equal(
        output_tensor.memory_config(), golden_tensor.memory_config()
    )
    assert all_close_check(output_tensor, golden_tensor)


# ------------------------------------------------------------
# Composite ops
# ------------------------------------------------------------
def cosh(input_tensor):
    e_pos_x = ttnn.exp(input_tensor)
    e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
    nr_term = ttnn.add(e_pos_x, e_neg_x)
    output = ttnn.multiply(nr_term, 0.5)
    return output


def sinh(input_tensor):
    e_pos_x = ttnn.exp(input_tensor)
    e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
    nr_term = ttnn.subtract(e_pos_x, e_neg_x)
    output = ttnn.multiply(nr_term, 0.5)
    return output


def mul_add(input_tensor_a, input_tensor_b, input_tensor_c):
    matmul_result = ttnn.multiply(input_tensor_b, input_tensor_c)
    output = ttnn.add(matmul_result, input_tensor_a)
    return output


@pytest.mark.parametrize("h , w, max_grid", COMMON_SHAPE_GRID_PARAMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_l1(device, h, w, max_grid, dtype, op):
    num_inputs = 1
    if op is mul_add:
        num_inputs = 3
    if op is mul_add and (h, w) == (256, 512) and dtype is torch.bfloat16:
        pytest.xfail("OOM error.")
    run_op_test(
        device, h, w, max_grid, dtype, op, num_inputs, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w", COMMON_SHAPE_PARAMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
@pytest.mark.skip(
    reason="Seeing failure with DRAM composite ops: /tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:1374: Unsupported memory space"
)
def test_composite_ops_dram(device, h, w, dtype, op):
    for h, w in COMMON_SHAPE_PARAMS:
        for dtype in [torch.bfloat16, torch.float32]:
            for op in [cosh, sinh, mul_add]:
                num_inputs = 1
                if op is mul_add:
                    num_inputs = 3
                run_op_test(
                    device,
                    h,
                    w,
                    max_grid=(0, 0),
                    dtype=dtype,
                    op=op,
                    num_inputs=num_inputs,
                    buffer_type=ttnn.BufferType.DRAM,
                )


passing_configs_l1 = {
    torch.bfloat16: [
        (512, 2048),
        (512, 4096),
        (512, 8192),
        (1024, 2048),
        (1024, 4096),
    ],
}


@pytest.mark.parametrize("seq_len", [2048, 4096, 8192, 16384, 32768, 65536])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_large_muladd_nice_seq_len_jit_l1(device, seq_len, hidden_dim, dtype):
    # most test cases fail
    if (hidden_dim, seq_len) not in passing_configs_l1.get(dtype, []):
        pytest.xfail(
            f"Most large L1 configs fail. Skipping ({hidden_dim}, {seq_len}) with dtype {dtype}."
        )

    max_grid = (7, 7)

    A = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)
    B = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)
    C = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("seq_len", [2048, 4096, 8192, 16384, 32768, 65536])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skip(
    reason="Seeing failure with DRAM composite ops: /tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:1374: Unsupported memory space"
)
def test_large_muladd_nice_seq_len_jit_dram(device, seq_len, hidden_dim, dtype):

    A = create_dram_tensor(device, seq_len, hidden_dim, dtype)
    B = create_dram_tensor(device, seq_len, hidden_dim, dtype)
    C = create_dram_tensor(device, seq_len, hidden_dim, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=(0, 0))(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("h, w, max_grid", COMMON_SHAPE_GRID_PARAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.xfail(
    reason="Broadcasting requires either h or w to be 1, but sharded tensor must be at least 32 x 32. Assert error."
)
def test_muladd_broadcast_jit_l1(device, h, w, max_grid, dtype):

    if max_grid == (7, 7):
        pytest.skip(
            "Fatal error in /tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/api/tt-metalium/math.hpp:27, 'Divide by zero error in div_up'"
        )

    A = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    B = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    # broadcast C
    C = create_sharded_tile_tensor(device, 1, w, max_grid, dtype)

    print("A:", A.cpu().to_torch())
    print("A shape:", A.shape)
    print("B:", B.cpu().to_torch())
    print("B shape:", B.shape)
    print("C:", C.cpu().to_torch())
    print("C shape:", C.shape)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


@pytest.mark.parametrize(
    "h, w", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skip(
    reason="Seeing failure with DRAM composite ops: /tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:1374: Unsupported memory space"
)
def test_muladd_broadcast_jit_dram(device, h, w, dtype):

    max_grid = (0, 0)
    A = create_dram_tensor(device, h, w, dtype)
    B = create_dram_tensor(device, h, w, dtype)
    # broadcast C
    C = create_dram_tensor(device, 1, w, dtype)

    print("A:", A.cpu().to_torch())
    print("A shape:", A.shape)
    print("B:", B.cpu().to_torch())
    print("B shape:", B.shape)
    print("C:", C.cpu().to_torch())
    print("C shape:", C.shape)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)
