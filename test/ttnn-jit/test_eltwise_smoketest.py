# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest
from op_definitions import *
from utils import (
    _get_ttnn_op,
    all_close_check,
    memory_configs_equal,
    get_expected_block_sharded_memory_config,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)


DRAM_SHAPES = [
    (1024, 1024),
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
def test_unary_op_dram(device, shape, dtype, ttnn_dtype, op):
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
def test_unary_op_l1(device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op):
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
def test_bitwise_unary_op_dram(device, shape, dtype, op):
    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
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
def test_bitwise_unary_op_l1(device, shape, max_grid, shard_strategy, dtype, op):

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
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
def test_binary_ops_dram(device, shape, dtype, ttnn_dtype, op):
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
def test_binary_ops_l1(device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op):
    if op in [div, pow, eq, ne, gt, ge, lt, le] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")
    if op in [maximum, minimum, pow] and ttnn_dtype == ttnn.DataType.BFLOAT8_B:
        pytest.xfail("failing allclose for some shapes for bfp8")
    if op == div and ttnn_dtype == ttnn.DataType.BFLOAT8_B:
        pytest.skip("ttnn.div does not support BFLOAT8_B")

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
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
        (torch.bfloat16, None),
    ],
    ids=["bf16"],
)
@pytest.mark.parametrize(
    "op",
    [add, sub, mul],
)
def test_binary_ops_mixed_layouts(
    device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op
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
        compile_only=False,
    )(op)

    output = compiled_op(input0, input1)
    expected_memory_config = get_expected_block_sharded_memory_config(
        output.shape, device
    )
    assert memory_configs_equal(output.memory_config(), expected_memory_config)

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
def test_bitwise_binary_ops_dram(device, shape, dtype, op):
    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.DRAM,
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
def test_bitwise_binary_ops_l1(device, shape, max_grid, shard_strategy, dtype, op):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.L1,
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

    expected_memory_config = get_expected_block_sharded_memory_config(
        golden_result.shape, device
    )
    assert memory_configs_equal(interop_result.memory_config(), expected_memory_config)
    assert all_close_check(interop_result, golden_result)


# ------------------------------------------------------------
# ttnn.clamp tests
# ------------------------------------------------------------


def _clamp_min_max(input_tensor):
    return ttnn.clamp(input_tensor, min=-0.001, max=0.001)


def _clamp_min_only(input_tensor):
    return ttnn.clamp(input_tensor, min=-0.001)


def _clamp_max_only(input_tensor):
    return ttnn.clamp(input_tensor, max=0.001)


def _clamp_tensor_bounds(input_tensor, min_tensor, max_tensor):
    return ttnn.clamp(input_tensor, min=min_tensor, max=max_tensor)


@pytest.mark.parametrize("shape", [(32, 32)], ids=["(32,32)"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize("op", [_clamp_min_max, _clamp_min_only, _clamp_max_only])
def test_clamp_scalar(device, shape, dtype, buffer_type, op):
    max_grid = (0, 0)
    shard_strategy = ttnn.ShardStrategy.BLOCK
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=1,
        buffer_type=buffer_type,
        shard_strategy=shard_strategy,
    )


@pytest.mark.parametrize(
    "buffer_type, shape, max_grid, shard_strategy",
    [
        (ttnn.BufferType.DRAM, (32, 32), (0, 0), None),
        (ttnn.BufferType.L1, (32, 32), (0, 0), ttnn.ShardStrategy.BLOCK),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_clamp_tensor(device, buffer_type, shape, max_grid, shard_strategy, dtype):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        _clamp_tensor_bounds,
        num_inputs=3,
        buffer_type=buffer_type,
        shard_strategy=shard_strategy,
    )


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

    expected_memory_config = get_expected_block_sharded_memory_config(
        golden_result.shape, device
    )
    assert memory_configs_equal(interop_result.memory_config(), expected_memory_config)
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

    expected_memory_config = get_expected_block_sharded_memory_config(
        golden_result.shape, device
    )
    assert memory_configs_equal(interop_result.memory_config(), expected_memory_config)
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

    expected_memory_config = get_expected_block_sharded_memory_config(
        golden_result.shape, device
    )
    assert memory_configs_equal(interop_result.memory_config(), expected_memory_config)
    assert all_close_check(interop_result, golden_result)
