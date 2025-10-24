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

BLOCK_SHARDED_SHAPE_GRIDS = [
    (32, 32, (0, 0)),
    (32, 64, (0, 0)),
    (64, 64, (0, 0)),
    (128, 128, (0, 0)),
    (256, 256, (7, 7)),
    (512, 512, (7, 7)),
    (512, 1024, (7, 7)),
    (1024, 1024, (7, 7)),
    (1024, 2048, (7, 7)),
    # Ensure non-square grid dims are interpreted correctly.
    (64, 128, (3, 0)),
    (64, 128, (0, 1)),
    (96, 128, (3, 2)),
]

DRAM_INTERLEAVED_SHAPES = [
    (32, 32),
    (32, 64),
    (64, 32),
    (256, 256),
    (1024, 2048),
    (2048, 2048),
    (1024, 32),
    (32, 1024),
    (64, 1024),
    (1024, 64),
]


# ------------------------------------------------------------
# Unary ops
# ------------------------------------------------------------
def abs(input_tensor):
    return ttnn.abs(input_tensor)


def exp(input_tensor):
    return ttnn.exp(input_tensor)


def log(input_tensor):
    return ttnn.log(input_tensor)


def cos(input_tensor):
    return ttnn.cos(input_tensor)


def sin(input_tensor):
    return ttnn.sin(input_tensor)


def tan(input_tensor):
    return ttnn.tan(input_tensor)


def cbrt(input_tensor):
    return ttnn.cbrt(input_tensor)


def ceil(input_tensor):
    return ttnn.ceil(input_tensor)


def sign(input_tensor):
    return ttnn.sign(input_tensor)


def erf(input_tensor):
    return ttnn.erf(input_tensor)


def erfc(input_tensor):
    return ttnn.erfc(input_tensor)


def floor(input_tensor):
    return ttnn.floor(input_tensor)


def gelu(input_tensor):
    return ttnn.gelu(input_tensor)


def logical_not(input_tensor):
    return ttnn.logical_not(input_tensor)


def bitwise_not(input_tensor):
    return ttnn.bitwise_not(input_tensor)


def reciprocal(input_tensor):
    return ttnn.reciprocal(input_tensor)


def sqrt(input_tensor):
    return ttnn.sqrt(input_tensor)


def rsqrt(input_tensor):
    return ttnn.rsqrt(input_tensor)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "op",
    [
        abs,
        exp,
        log,
        cos,
        sin,
        ceil,
        floor,
        logical_not,
    ],
)
@pytest.mark.parametrize(
    "h , w",
    DRAM_INTERLEAVED_SHAPES,
)
@pytest.mark.parametrize("graph_capture", [True, False])
def test_unary_op_dram(device, h, w, dtype, op, graph_capture):
    if op in [log, ceil, floor, logical_not] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")

    max_grid = (0, 0)
    run_op_test(
        device,
        h,
        w,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=graph_capture,
    )


@pytest.mark.parametrize("h , w, max_grid", BLOCK_SHARDED_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "op",
    [
        abs,
        exp,
        log,
        cos,
        sin,
        ceil,
        floor,
        logical_not,
        # Not supported in TTIRToD2M:
        # gelu, reciprocal cbrt, sign, erf, erfc
        # Always fails allclose
        # tan, sqrt
    ],
)
@pytest.mark.parametrize("graph_capture", [True, False])
def test_unary_op_l1(device, h, w, max_grid, dtype, op, graph_capture):
    if op in [log, ceil, floor, rsqrt, logical_not] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")

    run_op_test(
        device,
        h,
        w,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=graph_capture,
    )


@pytest.mark.parametrize("dtype", [torch.int32])
@pytest.mark.parametrize(
    "op",
    [
        bitwise_not,
    ],
)
@pytest.mark.parametrize(
    "h , w",
    DRAM_INTERLEAVED_SHAPES,
)
@pytest.mark.parametrize("graph_capture", [True, False])
def test_bitwise_unary_op_dram(device, h, w, dtype, op, graph_capture):
    max_grid = (0, 0)
    run_op_test(
        device,
        h,
        w,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=graph_capture,
    )


@pytest.mark.parametrize("h , w, max_grid", BLOCK_SHARDED_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.int32])
@pytest.mark.parametrize(
    "op",
    [
        bitwise_not,
    ],
)
@pytest.mark.parametrize("graph_capture", [True, False])
def test_bitwise_unary_op_l1(device, h, w, max_grid, dtype, op, graph_capture):
    run_op_test(
        device,
        h,
        w,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=graph_capture,
    )


# ------------------------------------------------------------
# Binary ops
# ------------------------------------------------------------
def add(a, b):
    return ttnn.add(a, b)


def sub(a, b):
    return ttnn.subtract(a, b)


def mul(a, b):
    return ttnn.multiply(a, b)


def div(a, b):
    return ttnn.divide(a, b)


def logical_and(a, b):
    return ttnn.logical_and(a, b)


def logical_or(a, b):
    return ttnn.logical_or(a, b)


def logical_xor(a, b):
    return ttnn.logical_xor(a, b)


def bitwise_or(a, b):
    return ttnn.bitwise_or(a, b)


def bitwise_and(a, b):
    return ttnn.bitwise_and(a, b)


def bitwise_xor(a, b):
    return ttnn.bitwise_xor(a, b)


def remainder(a, b):
    return ttnn.remainder(a, b)


def pow(a, b):
    # Test pow operation.
    #
    # Background:
    # -----------
    # The pow operation had a naming mismatch issue:
    # - The Python ttnn API has: ttnn.pow(a, b)
    # - The MLIR dialect has: ttnn.pow_tensor (not ttnn.pow)
    #
    # Original Issue:
    # --------------
    # PR #5154 changed the test from ttnn.pow() to ttnn.pow_tensor(), but this failed
    # because ttnn.pow_tensor doesn't exist in the Python API. When computing the
    # golden result (which calls the function directly without JIT), it would error:
    #     AttributeError: module 'ttnn' has no attribute 'pow_tensor'
    #
    # The Fix:
    # --------
    # 1. Use ttnn.pow() in the test (which exists in Python API)
    # 2. Added mapping in graph compiler: "pow" -> "pow_tensor" MLIR op
    # 3. Added mapping in AST compiler: node.attr "pow" -> "pow_tensor" MLIR op
    #
    # Both compilers automatically map ttnn.pow -> ttnn.pow_tensor MLIR operation.
    #
    # Note: float32 tests may xfail due to numerical precision issues.
    return ttnn.pow(a, b)


def atan2(a, b):
    return ttnn.atan2(a, b)


def eq(a, b):
    return ttnn.eq(a, b)


def ne(a, b):
    return ttnn.ne(a, b)


def gt(a, b):
    return ttnn.gt(a, b)


def ge(a, b):
    return ttnn.ge(a, b)


def lt(a, b):
    return ttnn.lt(a, b)


def le(a, b):
    return ttnn.le(a, b)


@pytest.mark.parametrize("h , w, max_grid", BLOCK_SHARDED_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "op",
    [
        add,
        sub,
        mul,
        div,
        pow,
        # logical_and, logical_or, logical_xor,
        # bitwise_or, bitwise_and, bitwise_xor, # not a supported FPU op
        # Not supported in TTIRToD2M
        # remainder, atan2, eq, ne, gt, ge, lt, le
    ],
)
@pytest.mark.parametrize("graph_capture", [True, False])
def test_binary_ops(device, h, w, max_grid, dtype, op, graph_capture):
    if op == div:
        pytest.xfail("failing allclose for some shapes")
    if op == pow and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")

    run_op_test(
        device,
        h,
        w,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=graph_capture,
    )


@pytest.mark.parametrize("h , w", DRAM_INTERLEAVED_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "op",
    [
        add,
        sub,
        mul,
        div,
        pow,
        # logical_and, logical_or, logical_xor,
        # bitwise_or, bitwise_and, bitwise_xor, # not a supported FPU op
        # Not supported in TTIRToD2M
        # remainder, atan2, eq, ne, gt, ge, lt, le
    ],
)
def test_binary_ops_dram(device, h, w, dtype, op):
    max_grid = (0, 0)
    if op == div:
        pytest.xfail("failing allclose for some shapes")
    if op == pow and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")

    run_op_test(
        device,
        h,
        w,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.DRAM,
    )


# ------------------------------------------------------------
# Interop tests
# ------------------------------------------------------------


# JIT op -> ttnn unary op test
@pytest.mark.parametrize("h, w, max_grid", BLOCK_SHARDED_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "jit_op, ttnn_unary_op",
    [
        (abs, ttnn.exp),
        (exp, ttnn.abs),
        (sin, ttnn.cos),
        (cos, ttnn.sin),
    ],
)
def test_interop_jit_to_ttnn_unary_l1(
    device, h, w, max_grid, dtype, jit_op, ttnn_unary_op
):
    input_tensor = create_sharded_tile_tensor(device, h, w, max_grid, dtype)

    # jit path
    compiled_op = ttnn_jit.jit(debug=True, max_grid=max_grid)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_unary_op(jit_output)

    # golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_unary_op(golden_jit_output)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# 2 JIT ops -> TTNN binary op test
@pytest.mark.parametrize("h, w, max_grid", BLOCK_SHARDED_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "jit_op1, jit_op2, ttnn_binary_op",
    [
        (abs, exp, ttnn.add),
        (sin, cos, ttnn.multiply),
        (exp, log, ttnn.subtract),
    ],
)
def test_interop_two_jit_to_ttnn_binary_l1(
    device, h, w, max_grid, dtype, jit_op1, jit_op2, ttnn_binary_op
):
    if jit_op2 == log and dtype == torch.float32:
        pytest.xfail("Failing all_close, getting nan values mismatching with golden")

    input1 = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    input2 = create_sharded_tile_tensor(device, h, w, max_grid, dtype)

    # interop path
    compiled_op1 = ttnn_jit.jit(debug=True, max_grid=max_grid)(jit_op1)
    compiled_op2 = ttnn_jit.jit(debug=True, max_grid=max_grid)(jit_op2)
    jit_output1 = compiled_op1(input1)
    jit_output2 = compiled_op2(input2)
    interop_result = ttnn_binary_op(jit_output1, jit_output2)

    # golden path
    golden_jit_op1 = _get_ttnn_op(jit_op1) or jit_op1
    golden_jit_op2 = _get_ttnn_op(jit_op2) or jit_op2
    golden_output1 = golden_jit_op1(input1)
    golden_output2 = golden_jit_op2(input2)
    golden_result = ttnn_binary_op(golden_output1, golden_output2)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# JIT op + ttnn tensor -> ttnn binary op test
@pytest.mark.parametrize("h, w, max_grid", BLOCK_SHARDED_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "jit_op, ttnn_binary_op",
    [
        (abs, ttnn.add),
        (exp, ttnn.multiply),
        (sin, ttnn.subtract),
    ],
)
def test_interop_jit_and_ttnn_to_binary_l1(
    device, h, w, max_grid, dtype, jit_op, ttnn_binary_op
):
    input_tensor = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    ttnn_tensor = create_sharded_tile_tensor(device, h, w, max_grid, dtype)

    # interop path
    compiled_op = ttnn_jit.jit(debug=True, max_grid=max_grid)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_binary_op(jit_output, ttnn_tensor)

    # golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_binary_op(golden_jit_output, ttnn_tensor)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# JIT op -> ttnn unary op test (DRAM)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "jit_op, ttnn_unary_op",
    [
        (abs, ttnn.exp),
        (exp, ttnn.abs),
        (sin, ttnn.cos),
        (cos, ttnn.sin),
    ],
)
@pytest.mark.parametrize(
    "h, w",
    DRAM_INTERLEAVED_SHAPES,
)
def test_interop_jit_to_ttnn_unary_dram(device, h, w, dtype, jit_op, ttnn_unary_op):
    max_grid = (0, 0)
    input_tensor = create_dram_tensor(device, h, w, dtype)

    # Interop path
    compiled_op = ttnn_jit.jit(debug=True, max_grid=max_grid)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_unary_op(jit_output)

    # Golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_unary_op(golden_jit_output)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# 2 JIT ops -> ttnn binary op test (DRAM)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "jit_op1, jit_op2, ttnn_binary_op",
    [
        (abs, exp, ttnn.add),
        (sin, cos, ttnn.multiply),
        (exp, log, ttnn.subtract),
    ],
)
@pytest.mark.parametrize(
    "h, w",
    DRAM_INTERLEAVED_SHAPES,
)
def test_interop_two_jit_to_ttnn_binary_dram(
    device, h, w, dtype, jit_op1, jit_op2, ttnn_binary_op
):
    if jit_op2 == log and dtype == torch.float32:
        pytest.xfail("Failing all_close, getting nan values mismatching with golden")

    max_grid = (0, 0)
    input1 = create_dram_tensor(device, h, w, dtype)
    input2 = create_dram_tensor(device, h, w, dtype)

    # Interop path
    compiled_op1 = ttnn_jit.jit(debug=True, max_grid=max_grid)(jit_op1)
    compiled_op2 = ttnn_jit.jit(debug=True, max_grid=max_grid)(jit_op2)
    jit_output1 = compiled_op1(input1)
    jit_output2 = compiled_op2(input2)
    interop_result = ttnn_binary_op(jit_output1, jit_output2)

    # Golden path
    golden_jit_op1 = _get_ttnn_op(jit_op1) or jit_op1
    golden_jit_op2 = _get_ttnn_op(jit_op2) or jit_op2
    golden_output1 = golden_jit_op1(input1)
    golden_output2 = golden_jit_op2(input2)
    golden_result = ttnn_binary_op(golden_output1, golden_output2)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# JIT op + ttnn tensor -> ttnn binary op test (DRAM)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "jit_op, ttnn_binary_op",
    [
        (abs, ttnn.add),
        (exp, ttnn.multiply),
        (sin, ttnn.subtract),
    ],
)
@pytest.mark.parametrize(
    "h, w",
    DRAM_INTERLEAVED_SHAPES,
)
def test_interop_jit_and_ttnn_to_binary_dram(
    device, h, w, dtype, jit_op, ttnn_binary_op
):
    max_grid = (0, 0)
    input_tensor = create_dram_tensor(device, h, w, dtype)
    ttnn_tensor = create_dram_tensor(device, h, w, dtype)

    # Interop path
    compiled_op = ttnn_jit.jit(debug=True, max_grid=max_grid)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_binary_op(jit_output, ttnn_tensor)

    # Golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_binary_op(golden_jit_output, ttnn_tensor)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)
