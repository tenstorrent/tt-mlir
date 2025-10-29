# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import _get_ttnn_op

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


def create_dram_tensor(device, h, w, dtype):
    torch.manual_seed(0)
    torch_tensor = torch.randn((h, w), dtype=dtype)
    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    return ttnn.from_torch(
        torch_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def create_sharded_tile_tensor(device, h, w, max_grid, dtype):
    torch.manual_seed(0)
    torch_tensor = torch.randn((h, w), dtype=dtype)

    start_coord = ttnn.CoreCoord(0, 0)
    end_coord = ttnn.CoreCoord(max_grid[0], max_grid[1])
    core_range = ttnn.CoreRange(start_coord, end_coord)
    core_range_set = ttnn.CoreRangeSet([core_range])

    shard_shape_x = h if max_grid[0] == 0 else h // (max_grid[0] + 1)
    shard_shape_y = w if max_grid[1] == 0 else w // (max_grid[1] + 1)

    shard_spec = ttnn.ShardSpec(
        grid=core_range_set,
        shard_shape=[shard_shape_x, shard_shape_y],
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    return ttnn.from_torch(
        torch_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def all_close_check(interop_result, golden_result, atol=1e-1, rtol=1e-1):

    print("--------------------------------")
    print("Interop result:")
    print(interop_result)
    print("--------------------------------")
    print("Golden result:")
    print(golden_result)
    print("--------------------------------")

    all_close = torch.allclose(
        interop_result.cpu().to_torch(),
        golden_result.cpu().to_torch(),
        atol=atol,
        rtol=rtol,
    )
    print("all_close", all_close)
    assert all_close


def run_op_test(
    device, h, w, max_grid, dtype, op, num_inputs, buffer_type=ttnn.BufferType.L1, use_ttir_compiler=False
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

    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid, use_ttir_compiler=use_ttir_compiler)(op)
    output_tensor = op_jit(*inputs)
    golden_tensor = (golden_op or op)(*inputs)

    all_close_check(output_tensor, golden_tensor)


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
    ],
)
@pytest.mark.parametrize(
    "h , w",
    [
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (256, 256),
        (256, 512),
    ],
)
def test_unary_op_dram(device, h, w, dtype, op):
    if op in [log, ceil, floor] and dtype == torch.float32:
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
    )


@pytest.mark.parametrize("h , w, max_grid", COMMON_SHAPE_GRID_PARAMS)
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
        # Not supported in TTIRToD2M:
        # gelu, logical_not, reciprocal cbrt, sign, erf, erfc, bitwise_not
        # Always fails allclose
        # tan, sqrt
    ],
)
def test_unary_op_l1(device, h, w, max_grid, dtype, op):
    if op in [log, ceil, floor, rsqrt] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")
    if op == abs and max_grid == (0, 0) and dtype == torch.bfloat16:
        pytest.skip("already tested in test_unary_op_single_core_sweep")

    run_op_test(
        device, h, w, max_grid, dtype, op, num_inputs=1, buffer_type=ttnn.BufferType.L1
    )


# Note: anything bigger than 256x256 will fail allclose for bfloat16
@pytest.mark.parametrize(
    "h, w", [(h, w) for h in range(32, 256, 32) for w in range(32, 256, 32)]
)
def test_unary_op_single_core_sweep(device, h, w):
    run_op_test(
        device,
        h,
        w,
        (0, 0),
        torch.bfloat16,
        abs,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
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
    # Note: Use ttnn.pow() not ttnn.pow_tensor()
    # - ttnn.pow() exists in the Python API (used for golden result computation)
    # - ttnn.pow_tensor only exists in MLIR dialect
    # - Both compilers map ttnn.pow -> ttnn.pow_tensor MLIR operation automatically
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


@pytest.mark.parametrize("h , w, max_grid", COMMON_SHAPE_GRID_PARAMS)
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
def test_binary_ops(device, h, w, max_grid, dtype, op):
    if op == div:
        pytest.xfail("failing allclose for some shapes")
    if op == pow and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")

    run_op_test(
        device, h, w, max_grid, dtype, op, num_inputs=2, buffer_type=ttnn.BufferType.L1
    )


@pytest.mark.parametrize("h , w, max_grid", COMMON_SHAPE_GRID_PARAMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pow_ast_compiler(device, h, w, max_grid, dtype):
    """Test pow operation with AST compiler (use_ttir_compiler=True).
    
    This test explicitly verifies that the pow operation works with the AST-based
    compiler path, which is separate from the default graph-based compiler.
    
    Background:
    -----------
    The pow operation had a naming mismatch issue:
    - The Python ttnn API has: ttnn.pow(a, b)
    - The MLIR dialect has: ttnn.pow_tensor (not ttnn.pow)
    
    Original Issue:
    --------------
    PR #5154 changed the test from ttnn.pow() to ttnn.pow_tensor(), but this failed
    because ttnn.pow_tensor doesn't exist in the Python API. When computing the
    golden result (which calls the function directly without JIT), it would error:
        AttributeError: module 'ttnn' has no attribute 'pow_tensor'
    
    The Fix:
    --------
    1. Changed test back to use ttnn.pow() (which exists in Python API)
    2. Added mapping in graph compiler: "pow" -> "pow_tensor" MLIR op
    3. Added mapping in AST compiler: node.attr "pow" -> "pow_tensor" MLIR op
    
    Why This Test Exists:
    --------------------
    The AST compiler (use_ttir_compiler=True) has a completely different code path
    from the graph compiler (default). It parses Python source code directly using
    the ast module, so we need explicit tests to ensure the pow -> pow_tensor
    mapping works correctly in this compilation path.
    
    Note: float32 tests are expected to xfail due to numerical precision issues.
    """
    if dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes with float32")
    
    run_op_test(
        device, h, w, max_grid, dtype, pow, num_inputs=2, buffer_type=ttnn.BufferType.L1, use_ttir_compiler=True
    )


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


# ------------------------------------------------------------
# Control flow tests
# ------------------------------------------------------------
def if_else_branch(input_tensor, use_exp=True):
    """Test basic if/else branching."""
    if use_exp:
        output = ttnn.exp(input_tensor)
    else:
        output = ttnn.log(input_tensor)
    return output


def nested_if(input_tensor, mode=0):
    """Test nested if statements."""
    if mode == 0:
        output = ttnn.abs(input_tensor)
    else:
        if mode == 1:
            output = ttnn.exp(input_tensor)
        else:
            output = ttnn.sin(input_tensor)
    return output


def if_with_ops_before_after(input_tensor, apply_exp=True):
    """Test if statement with operations before and after."""
    temp = ttnn.abs(input_tensor)
    
    if apply_exp:
        temp = ttnn.exp(temp)
    else:
        temp = ttnn.sin(temp)
    
    output = ttnn.multiply(temp, 0.5)
    return output


def multiple_sequential_ifs(input_tensor, apply_exp=False, apply_cos=False):
    """Test multiple sequential if statements."""
    output = input_tensor
    
    if apply_exp:
        output = ttnn.exp(output)
    
    if apply_cos:
        output = ttnn.cos(output)
    
    return output


def for_loop_simple(input_tensor, iterations=3):
    """Test simple for loop."""
    output = input_tensor
    for _ in range(iterations):
        output = ttnn.add(output, input_tensor)
    return output


def for_loop_with_index(input_tensor, iterations=3):
    """Test for loop with index variable."""
    output = input_tensor
    for i in range(iterations):
        if i == 0:
            output = ttnn.multiply(output, 2.0)
        else:
            output = ttnn.add(output, input_tensor)
    return output


def nested_for_loops(input_tensor, outer=2, inner=2):
    """Test nested for loops."""
    output = input_tensor
    for _ in range(outer):
        for _ in range(inner):
            output = ttnn.multiply(output, 1.1)
    return output


def for_loop_with_if(input_tensor, iterations=4):
    """Test for loop with if statement inside."""
    output = input_tensor
    for i in range(iterations):
        if i % 2 == 0:
            output = ttnn.add(output, input_tensor)
        else:
            output = ttnn.multiply(output, 0.9)
    return output


def if_inside_for_multiple_branches(input_tensor, iterations=3):
    """Test for loop with multiple if branches inside."""
    output = input_tensor
    for i in range(iterations):
        if i == 0:
            output = ttnn.exp(output)
        elif i == 1:
            output = ttnn.sin(output)
        else:
            output = ttnn.cos(output)
    return output


@pytest.mark.parametrize("h , w, max_grid", COMMON_SHAPE_GRID_PARAMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh])
def test_composite_ops(device, h, w, max_grid, dtype, op):
    run_op_test(device, h, w, max_grid, dtype, op, 1, buffer_type=ttnn.BufferType.L1)


# ------------------------------------------------------------
# Control flow tests
# ------------------------------------------------------------
@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0)), (256, 256, (7, 7))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_exp", [True, False])
def test_if_else_branch(device, h, w, max_grid, dtype, use_exp):
    """Test basic if/else branching with different branches."""
    def op_wrapper(input_tensor):
        return if_else_branch(input_tensor, use_exp=use_exp)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)


@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0)), (256, 256, (7, 7))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_nested_if(device, h, w, max_grid, dtype, mode):
    """Test nested if statements with different modes."""
    def op_wrapper(input_tensor):
        return nested_if(input_tensor, mode=mode)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)


@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("apply_exp", [True, False])
def test_if_with_ops_before_after(device, h, w, max_grid, dtype, apply_exp):
    """Test if statement with operations before and after."""
    def op_wrapper(input_tensor):
        return if_with_ops_before_after(input_tensor, apply_exp=apply_exp)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)


@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("apply_exp, apply_cos", [(True, False), (False, True), (True, True)])
def test_multiple_sequential_ifs(device, h, w, max_grid, dtype, apply_exp, apply_cos):
    """Test multiple sequential if statements. Note: (False, False) case excluded as it performs no operations."""
    def op_wrapper(input_tensor):
        return multiple_sequential_ifs(input_tensor, apply_exp=apply_exp, apply_cos=apply_cos)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)


@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0)), (256, 256, (7, 7))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("iterations", [1, 2, 3, 5])
def test_for_loop_simple(device, h, w, max_grid, dtype, iterations):
    """Test simple for loop with different iteration counts."""
    def op_wrapper(input_tensor):
        return for_loop_simple(input_tensor, iterations=iterations)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)


@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_for_loop_with_index(device, h, w, max_grid, dtype, iterations):
    """Test for loop with index variable and if statement."""
    def op_wrapper(input_tensor):
        return for_loop_with_index(input_tensor, iterations=iterations)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)


@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("outer, inner", [(2, 2), (2, 3), (3, 2)])
def test_nested_for_loops(device, h, w, max_grid, dtype, outer, inner):
    """Test nested for loops."""
    def op_wrapper(input_tensor):
        return nested_for_loops(input_tensor, outer=outer, inner=inner)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)


@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("iterations", [2, 3, 4])
def test_for_loop_with_if(device, h, w, max_grid, dtype, iterations):
    """Test for loop with if statement inside."""
    def op_wrapper(input_tensor):
        return for_loop_with_if(input_tensor, iterations=iterations)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)


@pytest.mark.parametrize("h , w, max_grid", [(32, 32, (0, 0)), (64, 64, (0, 0))])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("iterations", [3, 4, 5])
def test_if_inside_for_multiple_branches(device, h, w, max_grid, dtype, iterations):
    """Test for loop with multiple if branches inside."""
    def op_wrapper(input_tensor):
        return if_inside_for_multiple_branches(input_tensor, iterations=iterations)
    
    run_op_test(device, h, w, max_grid, dtype, op_wrapper, 1, buffer_type=ttnn.BufferType.L1)

# ------------------------------------------------------------
# Interop tests
# ------------------------------------------------------------


# JIT op -> ttnn unary op test
@pytest.mark.parametrize("h, w, max_grid", COMMON_SHAPE_GRID_PARAMS)
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
    compiled_op = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_unary_op(jit_output)

    # golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_unary_op(golden_jit_output)

    all_close_check(interop_result, golden_result)


# 2 JIT ops -> TTNN binary op test
@pytest.mark.parametrize("h, w, max_grid", COMMON_SHAPE_GRID_PARAMS)
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
    compiled_op1 = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(jit_op1)
    compiled_op2 = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(jit_op2)
    jit_output1 = compiled_op1(input1)
    jit_output2 = compiled_op2(input2)
    interop_result = ttnn_binary_op(jit_output1, jit_output2)

    # golden path
    golden_jit_op1 = _get_ttnn_op(jit_op1) or jit_op1
    golden_jit_op2 = _get_ttnn_op(jit_op2) or jit_op2
    golden_output1 = golden_jit_op1(input1)
    golden_output2 = golden_jit_op2(input2)
    golden_result = ttnn_binary_op(golden_output1, golden_output2)

    all_close_check(interop_result, golden_result)


# JIT op + ttnn tensor -> ttnn binary op test
@pytest.mark.parametrize("h, w, max_grid", COMMON_SHAPE_GRID_PARAMS)
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
    compiled_op = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_binary_op(jit_output, ttnn_tensor)

    # golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_binary_op(golden_jit_output, ttnn_tensor)

    all_close_check(interop_result, golden_result)


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
    [
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (256, 256),
        (256, 512),
    ],
)
def test_interop_jit_to_ttnn_unary_dram(device, h, w, dtype, jit_op, ttnn_unary_op):
    max_grid = (0, 0)
    input_tensor = create_dram_tensor(device, h, w, dtype)

    # Interop path
    compiled_op = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_unary_op(jit_output)

    # Golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_unary_op(golden_jit_output)

    all_close_check(interop_result, golden_result)


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
    [
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (256, 256),
        (256, 512),
    ],
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
    compiled_op1 = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(jit_op1)
    compiled_op2 = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(jit_op2)
    jit_output1 = compiled_op1(input1)
    jit_output2 = compiled_op2(input2)
    interop_result = ttnn_binary_op(jit_output1, jit_output2)

    # Golden path
    golden_jit_op1 = _get_ttnn_op(jit_op1) or jit_op1
    golden_jit_op2 = _get_ttnn_op(jit_op2) or jit_op2
    golden_output1 = golden_jit_op1(input1)
    golden_output2 = golden_jit_op2(input2)
    golden_result = ttnn_binary_op(golden_output1, golden_output2)

    all_close_check(interop_result, golden_result)


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
    [
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (256, 256),
        (256, 512),
    ],
)
def test_interop_jit_and_ttnn_to_binary_dram(
    device, h, w, dtype, jit_op, ttnn_binary_op
):
    max_grid = (0, 0)
    input_tensor = create_dram_tensor(device, h, w, dtype)
    ttnn_tensor = create_dram_tensor(device, h, w, dtype)

    # Interop path
    compiled_op = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(jit_op)
    jit_output = compiled_op(input_tensor)
    interop_result = ttnn_binary_op(jit_output, ttnn_tensor)

    # Golden path
    golden_jit_op = _get_ttnn_op(jit_op) or jit_op
    golden_jit_output = golden_jit_op(input_tensor)
    golden_result = ttnn_binary_op(golden_jit_output, ttnn_tensor)

    all_close_check(interop_result, golden_result)
