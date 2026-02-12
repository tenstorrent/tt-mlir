# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest
from op_definitions import *
from ttnn_jit._src.utils import get_maximal_block_sharding_grid
from utils import (
    _get_ttnn_op,
    all_close_check,
    memory_configs_equal,
    get_expected_block_sharded_memory_config,
    get_core_grid_from_device,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)

BLOCK_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((512, 1024), (7, 7)),
    ((1024, 1024), (7, 7)),
    # Ensure non-square grid dims are interpreted correctly.
    ((64, 128), (0, 1)),
    ((96, 128), (3, 2)),
    # Include rank 3 and 4 tensors.
    ((8, 32, 32), (0, 0)),
    ((2, 128, 128), (3, 0)),
    ((4, 256, 256), (7, 7)),
    ((4, 4, 32, 32), (0, 0)),
    ((2, 4, 64, 128), (3, 0)),
    ((1, 1, 256, 256), (7, 7)),
]

HEIGHT_SHARDED_SHAPE_GRIDS = [
    ((256, 32), (3, 0)),
    ((256, 32), (3, 1)),
    ((512, 32), (3, 3)),
    ((1024, 32), (7, 1)),
    ((1024, 32), (7, 3)),
    ((256, 64), (0, 7)),
    ((2048, 128), (7, 7)),
    ((384, 32), (1, 5)),
    ((2, 192, 32), (1, 5)),
    ((2, 2, 96, 32), (1, 5)),
    ((2, 2, 512, 32), (7, 7)),
]

WIDTH_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((64, 256), (7, 0)),
    ((64, 256), (0, 7)),
    ((128, 2048), (7, 7)),
    ((32, 384), (0, 5)),
    ((32, 384), (1, 5)),
    ((2, 32, 384), (1, 5)),
    ((2, 2, 32, 384), (1, 5)),
    ((2, 1, 32, 2048), (7, 7)),
]

# Combined list for tests that need all shard strategies (e.g., interop, binary mixed)
SHARDED_SHAPE_GRID_LAYOUTS = (
    [
        (shape, grid, ttnn.ShardStrategy.BLOCK)
        for shape, grid in BLOCK_SHARDED_SHAPE_GRIDS
    ]
    + [
        (shape, grid, ttnn.ShardStrategy.HEIGHT)
        for shape, grid in HEIGHT_SHARDED_SHAPE_GRIDS
    ]
    + [
        (shape, grid, ttnn.ShardStrategy.WIDTH)
        for shape, grid in WIDTH_SHARDED_SHAPE_GRIDS
    ]
)

# Minimal layouts for operations with limited dtype support
_MINIMAL_SHARDED_SHAPE_GRID_LAYOUTS = [
    (
        BLOCK_SHARDED_SHAPE_GRIDS[-1][0],
        BLOCK_SHARDED_SHAPE_GRIDS[-1][1],
        ttnn.ShardStrategy.BLOCK,
    ),
    (
        HEIGHT_SHARDED_SHAPE_GRIDS[-1][0],
        HEIGHT_SHARDED_SHAPE_GRIDS[-1][1],
        ttnn.ShardStrategy.HEIGHT,
    ),
    (
        WIDTH_SHARDED_SHAPE_GRIDS[-1][0],
        WIDTH_SHARDED_SHAPE_GRIDS[-1][1],
        ttnn.ShardStrategy.WIDTH,
    ),
]

DRAM_INTERLEAVED_SHAPES = [
    ((32, 32)),
    ((32, 64)),
    ((64, 32)),
    ((2048, 2048)),
    ((64, 1024)),
    ((1024, 64)),
    # Include rank 3 and 4 tensors.
    ((16, 64, 64)),
    ((4, 8, 32, 32)),
]


@pytest.mark.parametrize(
    "dtype, ttnn_dtype",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.DataType.BFLOAT8_B),
    ],
    ids=["f32", "bfp8"],
)
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
        tanh,
        sigmoid,
        hardsigmoid,
        sqrt,
        tan,
    ],
)
@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
)
def test_unary_op_dram(device, shape, dtype, ttnn_dtype, op):
    if (
        op in [log, ceil, floor, sqrt, reciprocal, logical_not]
        and dtype == torch.float32
    ):
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


UNARY_L1_BLOCK_PARAMS = [
    (shape, grid, ttnn.ShardStrategy.BLOCK, dtype, ttnn_dtype)
    for shape, grid in BLOCK_SHARDED_SHAPE_GRIDS
    for dtype, ttnn_dtype in [
        (torch.float32, None),
        (torch.bfloat16, None),
        (torch.bfloat16, ttnn.DataType.BFLOAT8_B),
    ]
]

UNARY_L1_HEIGHT_PARAMS = [
    (shape, grid, ttnn.ShardStrategy.HEIGHT, torch.bfloat16, None)
    for shape, grid in HEIGHT_SHARDED_SHAPE_GRIDS
]

UNARY_L1_WIDTH_PARAMS = [
    (shape, grid, ttnn.ShardStrategy.WIDTH, torch.bfloat16, None)
    for shape, grid in WIDTH_SHARDED_SHAPE_GRIDS
]

UNARY_L1_ALL_PARAMS = (
    UNARY_L1_BLOCK_PARAMS + UNARY_L1_HEIGHT_PARAMS + UNARY_L1_WIDTH_PARAMS
)


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy, dtype, ttnn_dtype",
    UNARY_L1_ALL_PARAMS,
    ids=[
        f"shape_{shape}_grid_{grid}_{strategy.name}_{('f32' if dtype==torch.float32 else 'bfp8' if ttnn_dtype==ttnn.DataType.BFLOAT8_B else 'bf16')}"
        for shape, grid, strategy, dtype, ttnn_dtype in UNARY_L1_ALL_PARAMS
    ],
)
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
        tanh,
        sigmoid,
        hardsigmoid,
        sqrt,
        reciprocal,
        tan,
    ],
)
def test_unary_op_l1(device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op):
    if op in [log, ceil, floor, sqrt, rsqrt, logical_not] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")

    if op == reciprocal and (
        ttnn_dtype == ttnn.DataType.BFLOAT8_B or dtype == torch.float32
    ):
        pytest.xfail("reciprocal not supported for bfp8")

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
    "shape, max_grid, shard_strategy",
    _MINIMAL_SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{shard_strategy.name}"
        for shape, grid, shard_strategy in _MINIMAL_SHARDED_SHAPE_GRID_LAYOUTS
    ],
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
    [
        erf,
        erfc,
        sign,
        relu,
        silu,
        gelu,
        tan,
    ],
)
def test_unary_op_l1_minimal(
    device, shape, max_grid, shard_strategy, dtype, ttnn_dtype, op
):
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


@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize(
    "op",
    [
        bitwise_not,
    ],
)
@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
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
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{shard_strategy}"
        for shape, grid, shard_strategy in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize(
    "op",
    [
        bitwise_not,
    ],
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
    "shape, max_grid, shard_strategy",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{shard_strategy}"
        for shape, grid, shard_strategy in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "op",
    [
        add,
    ],
)
def test_binary_ops_mixed1(device, shape, max_grid, shard_strategy, dtype, op):
    input0 = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )
    input1 = create_dram_tensor(device, shape, dtype)
    op_jit = ttnn_jit.jit(
        debug=True,
    )(op)
    output_tensor = op_jit(input0, input1)

    expected_memory_config = get_expected_block_sharded_memory_config(
        output_tensor.shape, device
    )
    assert memory_configs_equal(output_tensor.memory_config(), expected_memory_config)
    golden_output = op(input0, input1)
    pcc = ttnn.pearson_correlation_coefficient(
        golden_output.cpu().to_torch(), output_tensor.cpu().to_torch()
    )
    print("pcc: ", pcc)
    # assert pcc > 0.99, f"PCC: {pcc} is less than 0.99"
    assert all_close_check(output_tensor, golden_output)


BINARY_L1_BLOCK_PARAMS = [
    (shape, grid, ttnn.ShardStrategy.BLOCK, dtype)
    for shape, grid in BLOCK_SHARDED_SHAPE_GRIDS
    for dtype in [torch.float32, torch.bfloat16]
]

BINARY_L1_HEIGHT_PARAMS = [
    (shape, grid, ttnn.ShardStrategy.HEIGHT, torch.bfloat16)
    for shape, grid in HEIGHT_SHARDED_SHAPE_GRIDS
]

BINARY_L1_WIDTH_PARAMS = [
    (shape, grid, ttnn.ShardStrategy.WIDTH, torch.bfloat16)
    for shape, grid in WIDTH_SHARDED_SHAPE_GRIDS
]

BINARY_L1_ALL_PARAMS = (
    BINARY_L1_BLOCK_PARAMS + BINARY_L1_HEIGHT_PARAMS + BINARY_L1_WIDTH_PARAMS
)


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy, dtype",
    BINARY_L1_ALL_PARAMS,
    ids=[
        f"shape_{shape}_grid_{grid}_{strategy.name}_{('f32' if dtype==torch.float32 else 'bf16')}"
        for shape, grid, strategy, dtype in BINARY_L1_ALL_PARAMS
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        add,
        sub,
        mul,
        div,
        pow,
        eq,
        ne,
        gt,
        ge,
        lt,
        le,
        maximum,
        minimum,
    ],
)
def test_binary_ops_l1(device, shape, max_grid, shard_strategy, dtype, op):
    compile_only = False
    if op == div:
        compile_only = True
    if op in [pow, eq, ne, gt, ge, lt, le] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
        compile_only=compile_only,
    )


@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "op",
    [
        add,
        sub,
        mul,
        div,
        pow,
        eq,
        ne,
        gt,
        ge,
        lt,
        le,
        maximum,
        minimum,
        # logical_and, logical_or, logical_xor,
        # remainder, atan2,
    ],
)
def test_binary_ops_dram(device, shape, dtype, op):
    max_grid = (0, 0)
    compile_only = False
    if op == div:
        compile_only = True
    if op in [pow, eq, ne, gt, ge, lt, le] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.DRAM,
        compile_only=compile_only,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{shard_strategy}"
        for shape, grid, shard_strategy in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize(
    "op",
    [
        bitwise_and,
        bitwise_or,
        bitwise_xor,
    ],
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


@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize(
    "op",
    [
        bitwise_and,
        bitwise_or,
        bitwise_xor,
    ],
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


# ------------------------------------------------------------
# Interop tests
# ------------------------------------------------------------


# JIT op -> ttnn unary op test
@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{shard_strategy}"
        for shape, grid, shard_strategy in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
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
    device, shape, max_grid, shard_strategy, dtype, jit_op, ttnn_unary_op
):
    input_tensor = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )

    # jit path
    compiled_op = ttnn_jit.jit(debug=True)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_unary_op(jit_output)

    # golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_unary_op(golden_jit_output)

    expected_memory_config = get_expected_block_sharded_memory_config(
        golden_result.shape, device
    )
    assert memory_configs_equal(interop_result.memory_config(), expected_memory_config)
    assert all_close_check(interop_result, golden_result)


# 2 JIT ops -> TTNN binary op test
@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{shard_strategy}"
        for shape, grid, shard_strategy in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "jit_op1, jit_op2, ttnn_binary_op",
    [
        (abs, exp, ttnn.add),
        (sin, cos, ttnn.multiply),
        (exp, log, ttnn.subtract),
    ],
)
def test_interop_two_jit_to_ttnn_binary_l1(
    device, shape, max_grid, shard_strategy, dtype, jit_op1, jit_op2, ttnn_binary_op
):
    if jit_op2 == log and dtype == torch.float32:
        pytest.xfail("Failing all_close, getting nan values mismatching with golden")

    input1 = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )
    input2 = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )

    # interop path
    compiled_op1 = ttnn_jit.jit(debug=True)(jit_op1)
    compiled_op2 = ttnn_jit.jit(debug=True)(jit_op2)
    jit_output1 = compiled_op1(input1)
    jit_output2 = compiled_op2(input2)
    interop_result = ttnn_binary_op(jit_output1, jit_output2)

    # golden path
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


# JIT op + ttnn tensor -> ttnn binary op test
@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{shard_strategy}"
        for shape, grid, shard_strategy in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "jit_op, ttnn_binary_op",
    [
        (abs, ttnn.add),
        (exp, ttnn.multiply),
        (sin, ttnn.subtract),
    ],
)
def test_interop_jit_and_ttnn_to_binary_l1(
    device, shape, max_grid, shard_strategy, dtype, jit_op, ttnn_binary_op
):
    input_tensor = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )
    ttnn_tensor = create_sharded_tile_tensor(
        device, shape, max_grid, dtype, shard_strategy=shard_strategy
    )

    # interop path
    compiled_op = ttnn_jit.jit(debug=True)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_binary_op(jit_output, ttnn_tensor)

    # golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_binary_op(golden_jit_output, ttnn_tensor)

    expected_memory_config = get_expected_block_sharded_memory_config(
        golden_result.shape, device
    )
    assert memory_configs_equal(interop_result.memory_config(), expected_memory_config)
    assert all_close_check(interop_result, golden_result)


# JIT op -> ttnn unary op test (DRAM)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
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
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
)
def test_interop_jit_to_ttnn_unary_dram(device, shape, dtype, jit_op, ttnn_unary_op):
    input_tensor = create_dram_tensor(device, shape, dtype)

    # Interop path
    compiled_op = ttnn_jit.jit(debug=True)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_unary_op(jit_output)

    # Golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_unary_op(golden_jit_output)

    expected_memory_config = get_expected_block_sharded_memory_config(
        golden_result.shape, device
    )
    assert memory_configs_equal(interop_result.memory_config(), expected_memory_config)
    assert all_close_check(interop_result, golden_result)


# 2 JIT ops -> ttnn binary op test (DRAM)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "jit_op1, jit_op2, ttnn_binary_op",
    [
        (abs, exp, ttnn.add),
        (sin, cos, ttnn.multiply),
        (exp, log, ttnn.subtract),
    ],
)
@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
)
def test_interop_two_jit_to_ttnn_binary_dram(
    device, shape, dtype, jit_op1, jit_op2, ttnn_binary_op
):
    if jit_op2 == log and dtype == torch.float32:
        pytest.xfail("Failing all_close, getting nan values mismatching with golden")

    input1 = create_dram_tensor(device, shape, dtype)
    input2 = create_dram_tensor(device, shape, dtype)

    # Interop path
    compiled_op1 = ttnn_jit.jit(debug=True)(jit_op1)
    compiled_op2 = ttnn_jit.jit(debug=True)(jit_op2)
    jit_output1 = compiled_op1(input1)
    jit_output2 = compiled_op2(input2)
    interop_result = ttnn_binary_op(jit_output1, jit_output2)

    # Golden path
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


# JIT op + ttnn tensor -> ttnn binary op test (DRAM)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "jit_op, ttnn_binary_op",
    [
        (abs, ttnn.add),
        (exp, ttnn.multiply),
        (sin, ttnn.subtract),
    ],
)
@pytest.mark.parametrize(
    "shape",
    DRAM_INTERLEAVED_SHAPES,
    ids=[f"{shape}" for shape in DRAM_INTERLEAVED_SHAPES],
)
def test_interop_jit_and_ttnn_to_binary_dram(
    device, shape, dtype, jit_op, ttnn_binary_op
):
    input_tensor = create_dram_tensor(device, shape, dtype)
    ttnn_tensor = create_dram_tensor(device, shape, dtype)

    # Interop path
    compiled_op = ttnn_jit.jit(debug=True)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_binary_op(jit_output, ttnn_tensor)

    # Golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_binary_op(golden_jit_output, ttnn_tensor)

    expected_memory_config = get_expected_block_sharded_memory_config(
        golden_result.shape, device
    )
    assert memory_configs_equal(interop_result.memory_config(), expected_memory_config)
    assert all_close_check(interop_result, golden_result)


@pytest.mark.parametrize(
    "op",
    [add],
)
def test_tracing_binary_ops(device, op):
    """Test binary operations in tracing mode."""
    max_grid = (0, 0)
    run_op_test(
        device,
        (64, 32),
        max_grid,
        torch.bfloat16,
        op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.L1,
    )
