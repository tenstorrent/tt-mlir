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


DRAM_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (512, 2048),
    (2, 512, 2048),
    (4, 4, 32, 32),
]

SHARD_SHAPES_GRIDS = [
    ((1024, 1024), (7, 7), ttnn.ShardStrategy.BLOCK),
    ((2048, 32), (7, 7), ttnn.ShardStrategy.HEIGHT),
    ((32, 2048), (7, 7), ttnn.ShardStrategy.WIDTH),
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


def tanh(input_tensor):
    return ttnn.tanh(input_tensor)


def ceil(input_tensor):
    return ttnn.ceil(input_tensor)


def floor(input_tensor):
    return ttnn.floor(input_tensor)


def sign(input_tensor):
    return ttnn.sign(input_tensor)


def erf(input_tensor):
    return ttnn.erf(input_tensor)


def erfc(input_tensor):
    return ttnn.erfc(input_tensor)


def gelu(input_tensor):
    return ttnn.gelu(input_tensor)


def relu(input_tensor):
    return ttnn.relu(input_tensor)


def silu(input_tensor):
    return ttnn.silu(input_tensor)


def logical_not(input_tensor):
    return ttnn.logical_not(input_tensor)


def bitwise_not(input_tensor):
    return ttnn.bitwise_not(input_tensor)


def reciprocal(input_tensor):
    return ttnn.reciprocal(input_tensor)


def sqrt(input_tensor):
    return ttnn.sqrt(input_tensor)


def sigmoid(input_tensor):
    return ttnn.sigmoid(input_tensor)


def hardsigmoid(input_tensor):
    return ttnn.hardsigmoid(input_tensor)


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


def pow(a, b):
    return ttnn.pow(a, b)


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


def maximum(a, b):
    return ttnn.maximum(a, b)


def minimum(a, b):
    return ttnn.minimum(a, b)


def bitwise_or(a, b):
    return ttnn.bitwise_or(a, b)


def bitwise_and(a, b):
    return ttnn.bitwise_and(a, b)


def bitwise_xor(a, b):
    return ttnn.bitwise_xor(a, b)


# ------------------------------------------------------------
# Unary ops
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    DRAM_SHAPES,
    ids=[f"{shape}" for shape in DRAM_SHAPES],
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
    "op",
    [
        abs,
        exp,
        log,
        cos,
        sin,
        tan,
        ceil,
        floor,
        tanh,
        sigmoid,
        hardsigmoid,
        sqrt,
        logical_not,
    ],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_unary_op_dram(device, shape, dtype, ttnn_dtype, op, graph_capture):
    if dtype == torch.float32 and shape == (2048, 2048):
        pytest.skip("Skipping large operation for float32")
    if op in [log, ceil, floor, sqrt, logical_not] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=graph_capture,
        ttnn_dtype=ttnn_dtype,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARD_SHAPES_GRIDS,
    ids=[f"{shape}_{strategy.name}" for shape, grid, strategy in SHARD_SHAPES_GRIDS],
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
    "op",
    [
        abs,
        exp,
        log,
        cos,
        sin,
        tan,
        ceil,
        floor,
        tanh,
        sigmoid,
        hardsigmoid,
        sqrt,
        reciprocal,
        logical_not,
        erf,
        erfc,
        sign,
        relu,
        silu,
        gelu,
    ],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_unary_op_l1(
    device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op, graph_capture
):
    if op in [log, ceil, floor, sqrt, logical_not] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")
    if op == reciprocal and (
        ttnn_dtype == ttnn.DataType.BFLOAT8_B or dtype == torch.float32
    ):
        pytest.xfail("reciprocal not supported for bfp8 or f32")

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=graph_capture,
        shard_strategy=shard_strategy,
        ttnn_dtype=ttnn_dtype,
    )


@pytest.mark.parametrize(
    "shape",
    DRAM_SHAPES,
    ids=[f"{shape}" for shape in DRAM_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize(
    "op",
    [bitwise_not],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_bitwise_unary_op_dram(device, shape, dtype, op, graph_capture):
    if shape == (2048, 2048):
        pytest.skip("Skipping large operation")

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=graph_capture,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARD_SHAPES_GRIDS,
    ids=[f"{shape}_{strategy.name}" for shape, grid, strategy in SHARD_SHAPES_GRIDS],
)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize(
    "op",
    [bitwise_not],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_bitwise_unary_op_l1(
    device, shape, max_grid, shard_strategy, dtype, op, graph_capture
):

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=graph_capture,
        shard_strategy=shard_strategy,
    )


@pytest.mark.parametrize(
    "shape",
    DRAM_SHAPES,
    ids=[f"{shape}" for shape in DRAM_SHAPES],
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
    "op",
    [add, sub, mul, div, pow, eq, ne, gt, ge, lt, le, maximum, minimum],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_binary_ops_dram(device, shape, dtype, ttnn_dtype, op, graph_capture):
    if dtype == torch.float32 and shape == (2048, 2048):
        pytest.skip("Skipping large operation for float32")
    if op in [pow, eq, ne, gt, ge, lt, le] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")

    max_grid = (0, 0)
    compile_only = True if op == div else False

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=graph_capture,
        ttnn_dtype=ttnn_dtype,
        compile_only=compile_only,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARD_SHAPES_GRIDS,
    ids=[f"{shape}_{strategy.name}" for shape, grid, strategy in SHARD_SHAPES_GRIDS],
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
    "op",
    [add, sub, mul, div, pow, eq, ne, gt, ge, lt, le, maximum, minimum],
)
@pytest.mark.parametrize("graph_capture", [True, False])
def test_binary_ops_l1(
    device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op, graph_capture
):
    if op in [pow, eq, ne, gt, ge, lt, le] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")
    if op == div and ttnn_dtype == ttnn.DataType.BFLOAT8_B:
        pytest.xfail("mysterious error msg")

    compile_only = True if op == div else False

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=graph_capture,
        shard_strategy=shard_strategy,
        ttnn_dtype=ttnn_dtype,
        compile_only=compile_only,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARD_SHAPES_GRIDS,
    ids=[f"{shape}_{strategy.name}" for shape, grid, strategy in SHARD_SHAPES_GRIDS],
)
@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.bfloat16, None),
    ],
    ids=["bf16"],
)
@pytest.mark.parametrize(
    "op",
    [add, sub, mul],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_binary_ops_mixed_layouts(
    device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op, graph_capture
):
    input0 = create_sharded_tile_tensor(
        device,
        shape,
        max_grid,
        dtype,
        shard_strategy=shard_strategy,
        ttnn_dtype=ttnn_dtype,
    )
    input1 = create_dram_tensor(device, shape, dtype, ttnn_dtype=ttnn_dtype)

    compiled_op = ttnn_jit.jit(
        debug=True,
        graph_capture=graph_capture,
        compile_only=False,
    )(op)

    output = compiled_op(input0, input1)
    assert memory_configs_equal(output.memory_config(), input0.memory_config())

    golden_output = op(input0, input1)
    assert all_close_check(output, golden_output)
    pcc = ttnn.pearson_correlation_coefficient(
        golden_output.cpu().to_torch(), output.cpu().to_torch()
    )
    assert pcc > 0.99, f"PCC: {pcc} is less than 0.99"


@pytest.mark.parametrize(
    "shape",
    DRAM_SHAPES,
    ids=[f"{shape}" for shape in DRAM_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize(
    "op",
    [bitwise_and, bitwise_or, bitwise_xor],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_bitwise_binary_ops_dram(device, shape, dtype, op, graph_capture):
    if shape == (2048, 2048):
        pytest.skip("Skipping large operation")

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=graph_capture,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARD_SHAPES_GRIDS,
    ids=[f"{shape}_{strategy.name}" for shape, grid, strategy in SHARD_SHAPES_GRIDS],
)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize(
    "op",
    [bitwise_and, bitwise_or, bitwise_xor],
)
@pytest.mark.parametrize("graph_capture", [False])
def test_bitwise_binary_ops_l1(
    device, shape, max_grid, shard_strategy, dtype, op, graph_capture
):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.L1,
        graph_capture=graph_capture,
        shard_strategy=shard_strategy,
    )


# ------------------------------------------------------------
# Interop tests
# ------------------------------------------------------------


# JIT op -> ttnn unary op test
@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    [SHARD_SHAPES_GRIDS[0]],  # Use only BLOCK for interop
    ids=[f"{shape}_BLOCK" for shape, grid, strategy in [SHARD_SHAPES_GRIDS[0]]],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "jit_op, ttnn_unary_op",
    [
        (abs, ttnn.exp),
        (sin, ttnn.cos),
    ],
)
def test_interop_jit_to_ttnn_unary_l1(
    device, shape, max_grid, shard_strategy, dtype, jit_op, ttnn_unary_op
):
    input_tensor = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )

    compiled_op = ttnn_jit.jit(debug=True)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_unary_op(jit_output)

    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_unary_op(golden_jit_output)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# 2 JIT ops -> TTNN binary op test
@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    [SHARD_SHAPES_GRIDS[0]],
    ids=[f"{shape}_BLOCK" for shape, grid, strategy in [SHARD_SHAPES_GRIDS[0]]],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "jit_op1, jit_op2, ttnn_binary_op",
    [
        (abs, exp, ttnn.add),
    ],
)
def test_interop_two_jit_to_ttnn_binary_l1(
    device, shape, max_grid, shard_strategy, dtype, jit_op1, jit_op2, ttnn_binary_op
):
    input1 = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )
    input2 = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )

    compiled_op1 = ttnn_jit.jit(debug=True)(jit_op1)
    compiled_op2 = ttnn_jit.jit(debug=True)(jit_op2)
    jit_output1 = compiled_op1(input1)
    jit_output2 = compiled_op2(input2)
    interop_result = ttnn_binary_op(jit_output1, jit_output2)

    golden_jit_op1 = _get_ttnn_op(jit_op1) or jit_op1
    golden_jit_op2 = _get_ttnn_op(jit_op2) or jit_op2
    golden_output1 = golden_jit_op1(input1)
    golden_output2 = golden_jit_op2(input2)
    golden_result = ttnn_binary_op(golden_output1, golden_output2)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# JIT op -> ttnn unary op test (DRAM)
@pytest.mark.parametrize(
    "shape",
    [DRAM_SHAPES[0]],  # Use one shape for interop
    ids=[f"{DRAM_SHAPES[0]}"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "jit_op, ttnn_unary_op",
    [
        (abs, ttnn.exp),
    ],
)
def test_interop_jit_to_ttnn_unary_dram(device, shape, dtype, jit_op, ttnn_unary_op):
    input_tensor = create_dram_tensor(device, shape, dtype)

    compiled_op = ttnn_jit.jit(debug=True)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_unary_op(jit_output)

    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_unary_op(golden_jit_output)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# 2 JIT ops -> ttnn binary op test (DRAM)
@pytest.mark.parametrize(
    "shape",
    [DRAM_SHAPES[0]],
    ids=[f"{DRAM_SHAPES[0]}"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "jit_op1, jit_op2, ttnn_binary_op",
    [
        (abs, exp, ttnn.add),
    ],
)
def test_interop_two_jit_to_ttnn_binary_dram(
    device, shape, dtype, jit_op1, jit_op2, ttnn_binary_op
):
    input1 = create_dram_tensor(device, shape, dtype)
    input2 = create_dram_tensor(device, shape, dtype)

    compiled_op1 = ttnn_jit.jit(debug=True)(jit_op1)
    compiled_op2 = ttnn_jit.jit(debug=True)(jit_op2)
    jit_output1 = compiled_op1(input1)
    jit_output2 = compiled_op2(input2)
    interop_result = ttnn_binary_op(jit_output1, jit_output2)

    golden_jit_op1 = _get_ttnn_op(jit_op1) or jit_op1
    golden_jit_op2 = _get_ttnn_op(jit_op2) or jit_op2
    golden_output1 = golden_jit_op1(input1)
    golden_output2 = golden_jit_op2(input2)
    golden_result = ttnn_binary_op(golden_output1, golden_output2)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)
