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
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((64, 64), (0, 0)),
    ((64, 128), (0, 0)),
    ((256, 256), (7, 7)),
    ((256, 512), (7, 7)),
    ((512, 512), (7, 7)),
    ((512, 1024), (7, 7)),
    ((1024, 1024), (7, 7)),
    ((2, 512, 2048), (7, 7)),
]

HEIGHT_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((256, 64), (7, 0)),
    ((256, 64), (0, 7)),
    ((2048, 128), (7, 7)),
    ((2, 192, 32), (1, 5)),
]

WIDTH_SHARDED_SHAPE_GRIDS = [
    ((32, 32), (0, 0)),
    ((32, 64), (0, 0)),
    ((64, 256), (7, 0)),
    ((64, 256), (0, 7)),
    ((128, 2048), (7, 7)),
    ((2, 32, 384), (1, 5)),
]

SHARDED_SHAPE_GRID_LAYOUTS = (
    [
        (shape, grid, ttnn.TensorMemoryLayout.BLOCK_SHARDED)
        for shape, grid in BLOCK_SHARDED_SHAPE_GRIDS
    ]
    + [
        (shape, grid, ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
        for shape, grid in HEIGHT_SHARDED_SHAPE_GRIDS
    ]
    + [
        (shape, grid, ttnn.TensorMemoryLayout.WIDTH_SHARDED)
        for shape, grid in WIDTH_SHARDED_SHAPE_GRIDS
    ]
)

DRAM_SHAPES = [(32, 32), (32, 64), (64, 64), (64, 128), (128, 128)]


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


@pytest.mark.parametrize(
    "shape, max_grid, memory_layout",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{layout}"
        for shape, grid, layout in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_l1(device, shape, max_grid, dtype, op, memory_layout):
    num_inputs = 1
    if op is mul_add:
        num_inputs = 3
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=memory_layout,
    )


@pytest.mark.parametrize("shape", DRAM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_dram(device, shape, dtype, op):
    num_inputs = 1
    if op is mul_add:
        num_inputs = 3
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=op,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
    )


PASSING_LARGE_SHAPES_DTYPES_L1 = [
    ((2048, 512), torch.bfloat16),
    ((4096, 512), torch.bfloat16),
    ((8192, 512), torch.bfloat16),
    ((2048, 1024), torch.bfloat16),
]


@pytest.mark.parametrize("shape, dtype", PASSING_LARGE_SHAPES_DTYPES_L1)
def test_large_shapes_muladd_l1(device, shape, dtype):

    num_inputs = 3

    run_op_test(
        device,
        shape,
        max_grid=(7, 7),
        dtype=dtype,
        op=mul_add,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.L1,
    )


PASSING_LARGE_SHAPES_DTYPES_DRAM = [
    ((2048, 512), torch.float32),
    ((4096, 512), torch.float32),
    ((8192, 512), torch.float32),
    ((2048, 1024), torch.float32),
    ((4096, 1024), torch.float32),
    ((2048, 2048), torch.float32),
    ((1024, 4096), torch.float32),
    ((2048, 512), torch.bfloat16),
    ((4096, 512), torch.bfloat16),
    ((8192, 512), torch.bfloat16),
    ((16384, 512), torch.bfloat16),
    ((2048, 1024), torch.bfloat16),
    ((4096, 1024), torch.bfloat16),
    ((8192, 1024), torch.bfloat16),
    ((4096, 2048), torch.bfloat16),
]


@pytest.mark.parametrize("shape, dtype", PASSING_LARGE_SHAPES_DTYPES_DRAM)
def test_large_shapes_muladd_dram(device, shape, dtype):

    num_inputs = 3

    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=mul_add,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
    )


@pytest.mark.parametrize("shape, max_grid", BLOCK_SHARDED_SHAPE_GRIDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.xfail(
    reason="Broadcasting requires either h or w to be 1, but sharded tensor must be at least 32 x 32. Assert error."
)
def test_muladd_broadcast_jit_l1(device, shape, max_grid, dtype):

    if max_grid == (7, 7):
        pytest.skip(
            "Fatal error in /tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/api/tt-metalium/math.hpp:27, 'Divide by zero error in div_up'"
        )

    A = create_sharded_tile_tensor(device, shape, max_grid, dtype)
    B = create_sharded_tile_tensor(device, shape, max_grid, dtype)
    # broadcast C
    C = create_sharded_tile_tensor(device, 1, shape[1], max_grid, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


@pytest.mark.parametrize(
    "shape", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.xfail(reason="All tests failing allclose.")
def test_muladd_broadcast_jit_dram(device, shape, dtype):

    max_grid = (0, 0)
    A = create_dram_tensor(device, shape, dtype)
    B = create_dram_tensor(device, shape, dtype)
    # broadcast C
    C = create_dram_tensor(device, 1, shape[1], dtype)

    # JIT path
    op_jit = ttnn_jit.jit(debug=True)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )
    assert all_close_check(interop_result, golden_result)


# ------------------------------------------------------------
# Special functions
# ------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.xfail(reason="D2M Does not support reciprocal op")
def test_digamma_dram(device, shape, dtype):
    """Test digamma function (derivative of log gamma)"""

    def digamma_func(input_tensor):
        return ttnn.digamma(input_tensor)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        digamma_func,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_complex_composite_dram(device, shape, dtype):
    """Test complex composite operation with multiple steps"""

    def complex_op(a, b):
        # (a + b) * (a - b) = a^2 - b^2
        sum_ab = ttnn.add(a, b)
        diff_ab = ttnn.subtract(a, b)
        return ttnn.multiply(sum_ab, diff_ab)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        complex_op,
        num_inputs=2,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nested_composite_dram(device, shape, dtype):
    """Test nested composite operations"""

    def nested_op(a, b, c):
        # ((a + b) * c) + a
        sum_ab = ttnn.add(a, b)
        prod_abc = ttnn.multiply(sum_ab, c)
        return ttnn.add(prod_abc, a)

    max_grid = (0, 0)
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        nested_op,
        num_inputs=3,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=True,
    )


# ------------------------------------------------------------
# Fusion Tests
# ------------------------------------------------------------

# Chain can be made arbitrarily long when using JIT (as long as the 
# instructions fit on the tensix I-SRAM) and executed as one compute
# kernel. With a sufficiently long chain and large enough input size,
# regular ttnn will run out of memory while ttnn.jit will not.
def long_unary_chain(input_tensor_a):
    res_0 = ttnn.abs(input_tensor_a)
    res_1 = ttnn.sin(res_0)
    res_2 = ttnn.neg(res_1)
    res_3 = ttnn.exp(res_2)

    res_4 = ttnn.abs(res_3)
    res_5 = ttnn.cos(res_4)
    res_6 = ttnn.neg(res_5)
    res_7 = ttnn.exp(res_6)

    return res_7
    # replace the above return with the code below
    # to see shapes/grids where ttnn runs out of memory 
    # but ttnn.jit, through fusion, does not.
    
    # res_8 = ttnn.neg(res_7)
    # return res_8


@pytest.mark.parametrize(
    "shape, max_grid, memory_layout",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{layout}"
        for shape, grid, layout in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_long_unary_chain_l1(device, shape, max_grid, dtype, memory_layout):
    num_inputs = 1

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        long_unary_chain,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=memory_layout,
    )


@pytest.mark.parametrize("shape", DRAM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_long_unary_chain_dram(device, shape, dtype):
    num_inputs = 1

    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=long_unary_chain,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
    )


### ------------------------------------------------------------------------ ###


# Example of fusing 2 unary op chains that eventually join through a binary op.
# Unary chains can be made arbitrarily long with jit.
def join_unary_chains(in0, in1):
    chain_0_0 = ttnn.abs(in0)
    chain_0_1 = ttnn.exp(chain_0_0)
    chain_0_2 = ttnn.neg(chain_0_1)

    chain_1_0 = ttnn.abs(in1)
    chain_1_1 = ttnn.exp(chain_1_0)
    chain_1_2 = ttnn.neg(chain_1_1)

    return ttnn.add(chain_0_2, chain_1_2)


@pytest.mark.parametrize(
    "shape, max_grid, memory_layout",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{layout}"
        for shape, grid, layout in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_join_unary_chains_l1(device, shape, max_grid, dtype, memory_layout):
    num_inputs = 2

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        join_unary_chains,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=memory_layout,
        graph_capture=False
    )


@pytest.mark.parametrize("shape", DRAM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_join_unary_chains_dram(device, shape, dtype):
    num_inputs = 2

    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=join_unary_chains,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=False
    )


### ------------------------------------------------------------------------ ###


# The largest BALANCED op tree that can be fused into one d2m.generic and fully DST fused
# when in 32b DST mode.
def add_tree_7_to_1(
    in0, in1, in2, in3, 
    in4,in5, in6
):
    add_0_0 = ttnn.add(in0, in1)
    add_0_1 = ttnn.add(in2, in3)
    add_0_2 = ttnn.add(in4, in5)

    add_1_0 = ttnn.add(add_0_0, add_0_1)
    add_1_1 = ttnn.add(add_0_2, in6)

    add_2_0 = ttnn.add(add_1_0, add_1_1)

    return add_2_0


@pytest.mark.parametrize(
    "shape, max_grid, memory_layout",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{layout}"
        for shape, grid, layout in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_tree_7_to_1_l1(device, shape, max_grid, dtype, memory_layout):
    num_inputs = 7

    if shape == (2, 512, 2048):
        pytest.xfail(
            "add_tree_7_to_1 runs out of memory in regular TTNN path"
        )

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        add_tree_7_to_1,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=memory_layout,
        graph_capture=False
    )


@pytest.mark.parametrize("shape", DRAM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_tree_7_to_1_dram(device, shape, dtype):
    num_inputs = 7

    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=add_tree_7_to_1,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=False
    )


### ------------------------------------------------------------------------ ###


# The largest BALANCED op tree that can be fused into one d2m.generic and fully DST fused
# when in 16b DST mode. Works with 32b as well but results in more ops/kernels
def add_tree_31_to_1(
    in0, in1, in2, in3, in4, in5, in6, in7,
    in8, in9, in10, in11, in12, in13, in14, in15,
    in16, in17, in18, in19, in20, in21, in22, in23,
    in24, in25, in26, in27, in28, in29, in30
):
    # Level 0: 15 pairs + 1 unpaired (31 -> 16)
    add_0_0 = ttnn.add(in0, in1)
    add_0_1 = ttnn.add(in2, in3)
    add_0_2 = ttnn.add(in4, in5)
    add_0_3 = ttnn.add(in6, in7)
    add_0_4 = ttnn.add(in8, in9)
    add_0_5 = ttnn.add(in10, in11)
    add_0_6 = ttnn.add(in12, in13)
    add_0_7 = ttnn.add(in14, in15)
    add_0_8 = ttnn.add(in16, in17)
    add_0_9 = ttnn.add(in18, in19)
    add_0_10 = ttnn.add(in20, in21)
    add_0_11 = ttnn.add(in22, in23)
    add_0_12 = ttnn.add(in24, in25)
    add_0_13 = ttnn.add(in26, in27)
    add_0_14 = ttnn.add(in28, in29)

    # Level 1: 8 pairs (16 -> 8)
    add_1_0 = ttnn.add(add_0_0, add_0_1)
    add_1_1 = ttnn.add(add_0_2, add_0_3)
    add_1_2 = ttnn.add(add_0_4, add_0_5)
    add_1_3 = ttnn.add(add_0_6, add_0_7)
    add_1_4 = ttnn.add(add_0_8, add_0_9)
    add_1_5 = ttnn.add(add_0_10, add_0_11)
    add_1_6 = ttnn.add(add_0_12, add_0_13)
    add_1_7 = ttnn.add(add_0_14, in30)

    # Level 2: 4 pairs (8 -> 4)
    add_2_0 = ttnn.add(add_1_0, add_1_1)
    add_2_1 = ttnn.add(add_1_2, add_1_3)
    add_2_2 = ttnn.add(add_1_4, add_1_5)
    add_2_3 = ttnn.add(add_1_6, add_1_7)

    # Level 3: 2 pairs (4 -> 2)
    add_3_0 = ttnn.add(add_2_0, add_2_1)
    add_3_1 = ttnn.add(add_2_2, add_2_3)

    # Level 4: 1 pair (2 -> 1)
    add_4_0 = ttnn.add(add_3_0, add_3_1)

    return add_4_0


@pytest.mark.parametrize(
    "shape, max_grid, memory_layout",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{layout}"
        for shape, grid, layout in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_add_tree_31_to_1_l1(device, shape, max_grid, dtype, memory_layout):
    num_inputs = 31

    size_limit = 256 * 256

    elements = 1
    for dim in shape:
        elements *= dim
        
    if elements > size_limit:
        pytest.xfail(
            "add_tree_31_to_1 runs out of memory"
        )
    
    if shape == (2, 32, 384):
        pytest.xfail(
            "add_tree_31_to_1 fails all close"
        )

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        add_tree_31_to_1,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=memory_layout,
        graph_capture=False
    )


@pytest.mark.parametrize("shape", DRAM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_tree_31_to_1_dram(device, shape, dtype):
    num_inputs = 31

    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=add_tree_31_to_1,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=False
    )


### ------------------------------------------------------------------------ ###


# A ladder pattern where each add output is summed with the next input sequentially.
# Unlike the balanced tree, this creates a long dependency chain.
def adder_ladder_31(
    in0, in1, in2, in3, in4, in5, in6, in7,
    in8, in9, in10, in11, in12, in13, in14, in15,
    in16, in17, in18, in19, in20, in21, in22, in23,
    in24, in25, in26, in27, in28, in29, in30
):
    add_0 = ttnn.add(in0, in1)
    add_1 = ttnn.add(add_0, in2)
    add_2 = ttnn.add(add_1, in3)
    add_3 = ttnn.add(add_2, in4)
    add_4 = ttnn.add(add_3, in5)
    add_5 = ttnn.add(add_4, in6)
    add_6 = ttnn.add(add_5, in7)
    add_7 = ttnn.add(add_6, in8)
    add_8 = ttnn.add(add_7, in9)
    add_9 = ttnn.add(add_8, in10)
    add_10 = ttnn.add(add_9, in11)
    add_11 = ttnn.add(add_10, in12)
    add_12 = ttnn.add(add_11, in13)
    add_13 = ttnn.add(add_12, in14)
    add_14 = ttnn.add(add_13, in15)
    add_15 = ttnn.add(add_14, in16)
    add_16 = ttnn.add(add_15, in17)
    add_17 = ttnn.add(add_16, in18)
    add_18 = ttnn.add(add_17, in19)
    add_19 = ttnn.add(add_18, in20)
    add_20 = ttnn.add(add_19, in21)
    add_21 = ttnn.add(add_20, in22)
    add_22 = ttnn.add(add_21, in23)
    add_23 = ttnn.add(add_22, in24)
    add_24 = ttnn.add(add_23, in25)
    add_25 = ttnn.add(add_24, in26)
    add_26 = ttnn.add(add_25, in27)
    add_27 = ttnn.add(add_26, in28)
    add_28 = ttnn.add(add_27, in29)
    add_29 = ttnn.add(add_28, in30)

    return add_29


@pytest.mark.parametrize(
    "shape, max_grid, memory_layout",
    SHARDED_SHAPE_GRID_LAYOUTS,
    ids=[
        f"shape_{shape}_grid_{grid}_{layout}"
        for shape, grid, layout in SHARDED_SHAPE_GRID_LAYOUTS
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_adder_ladder_31_l1(device, shape, max_grid, dtype, memory_layout):
    num_inputs = 31

    if dtype is torch.bfloat16:
        pytest.xfail(
            "adder_ladder_31 fails all close"
        )

    size_limit = 256 * 256
    elements = 1

    for dim in shape:
        elements *= dim
        
    if elements > size_limit or (shape == (64, 128) and max_grid == (0,0)):
        pytest.xfail(
            "adder_ladder_31 runs out of memory"
        )

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        adder_ladder_31,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=memory_layout,
        graph_capture=False
    )


@pytest.mark.parametrize("shape", DRAM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_adder_ladder_31_dram(device, shape, dtype):
    num_inputs = 31

    if dtype is torch.bfloat16:
        pytest.xfail(
            "adder_ladder_31 fails all close"
        )

    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=adder_ladder_31,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
        graph_capture=False
    )