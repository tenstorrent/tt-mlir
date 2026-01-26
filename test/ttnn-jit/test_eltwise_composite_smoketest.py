# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    all_close_check,
    memory_configs_equal,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)
from op_definitions import cosh, sinh, mul_add


DRAM_SHAPES = [
    (64, 64),
    (256, 256),
    (512, 512),
    (256, 1024),
    (2, 512, 2048),
    (1, 4, 256, 1024),
]


SHARD_SHAPE_GRIDS = [
    ((512, 512), (7, 7), ttnn.ShardStrategy.BLOCK),
    ((2048, 128), (7, 7), ttnn.ShardStrategy.HEIGHT),
    ((128, 2048), (7, 7), ttnn.ShardStrategy.WIDTH),
]


@pytest.mark.parametrize(
    "shape",
    DRAM_SHAPES,
    ids=[f"{shape}" for shape in DRAM_SHAPES],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_dram(device, shape, dtype, op):
    num_inputs = 3 if op is mul_add else 1

    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=op,
        num_inputs=num_inputs,
        buffer_type=ttnn.BufferType.DRAM,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    SHARD_SHAPE_GRIDS,
    ids=[f"{shape}_{strategy.name}" for shape, grid, strategy in SHARD_SHAPE_GRIDS],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("op", [cosh, sinh, mul_add])
def test_composite_ops_l1(device, shape, max_grid, shard_strategy, dtype, op):
    num_inputs = 3 if op is mul_add else 1

    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        op,
        num_inputs,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
    )


# ------------------------------------------------------------
# Special functions
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(64, 64), (128, 128)],
    ids=["64x64", "128x128"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
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
    )


@pytest.mark.parametrize(
    "shape",
    [(64, 64), (128, 128)],
    ids=["64x64", "128x128"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
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
    )


# ------------------------------------------------------------
# Fusion Tests
# ------------------------------------------------------------
# Chain can be made arbitrarily long when using JIT (as long as the
# instructions fit on the tensix I-SRAM) and executed as one compute
# kernel.
def long_unary_chain(input_tensor_a):
    res_0 = ttnn.abs(input_tensor_a)
    res_1 = ttnn.sin(res_0)
    res_2 = ttnn.neg(res_1)
    res_3 = ttnn.exp(res_2)
    res_4 = ttnn.abs(res_3)
    res_5 = ttnn.cos(res_4)
    res_6 = ttnn.neg(res_5)
    res_7 = ttnn.exp(res_6)
    res_8 = ttnn.neg(res_7)
    return res_8


@pytest.mark.parametrize(
    "shape",
    [(64, 64), (128, 128)],
    ids=["64x64", "128x128"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
def test_long_unary_chain_dram(device, shape, dtype):
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=long_unary_chain,
        num_inputs=1,
        buffer_type=ttnn.BufferType.DRAM,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    [SHARD_SHAPE_GRIDS[0]],  # Use only BLOCK for fusion tests
    ids=[f"{shape}_BLOCK" for shape, grid, strategy in [SHARD_SHAPE_GRIDS[0]]],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
def test_long_unary_chain_l1(device, shape, max_grid, shard_strategy, dtype):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        long_unary_chain,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
    )


### ------------------------------------------------------------------------ ###


# Example of fusing 2 unary op chains that eventually join through a binary op.
# Unary chains can be arbitrarily long and still be fused when using jit.
def join_unary_chains(in0, in1):
    chain_0_0 = ttnn.abs(in0)
    chain_0_1 = ttnn.exp(chain_0_0)
    chain_0_2 = ttnn.neg(chain_0_1)

    chain_1_0 = ttnn.abs(in1)
    chain_1_1 = ttnn.exp(chain_1_0)
    chain_1_2 = ttnn.neg(chain_1_1)

    return ttnn.add(chain_0_2, chain_1_2)


@pytest.mark.parametrize(
    "shape",
    [(64, 64), (128, 128)],
    ids=["64x64", "128x128"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
def test_join_unary_chains_dram(device, shape, dtype):
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=join_unary_chains,
        num_inputs=2,
        buffer_type=ttnn.BufferType.DRAM,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    [SHARD_SHAPE_GRIDS[0]],
    ids=[f"{shape}_BLOCK" for shape, grid, strategy in [SHARD_SHAPE_GRIDS[0]]],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
def test_join_unary_chains_l1(device, shape, max_grid, shard_strategy, dtype):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        join_unary_chains,
        num_inputs=2,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
    )


### ------------------------------------------------------------------------ ###


# The largest BALANCED op tree that can be fused into one d2m.generic and fully DST fused
# when in 32b DST mode.
def add_tree_7_to_1(in0, in1, in2, in3, in4, in5, in6):
    add_0_0 = ttnn.add(in0, in1)
    add_0_1 = ttnn.add(in2, in3)
    add_0_2 = ttnn.add(in4, in5)

    add_1_0 = ttnn.add(add_0_0, add_0_1)
    add_1_1 = ttnn.add(add_0_2, in6)

    add_2_0 = ttnn.add(add_1_0, add_1_1)

    return add_2_0


@pytest.mark.parametrize(
    "shape",
    [(64, 64), (128, 128)],
    ids=["64x64", "128x128"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
def test_add_tree_7_to_1_dram(device, shape, dtype):
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=add_tree_7_to_1,
        num_inputs=7,
        buffer_type=ttnn.BufferType.DRAM,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    [SHARD_SHAPE_GRIDS[0]],
    ids=[f"{shape}_BLOCK" for shape, grid, strategy in [SHARD_SHAPE_GRIDS[0]]],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
def test_add_tree_7_to_1_l1(device, shape, max_grid, shard_strategy, dtype):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        add_tree_7_to_1,
        num_inputs=7,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
    )


### ------------------------------------------------------------------------ ###


# The largest BALANCED op tree that can be fused into one d2m.generic and fully DST fused
# when in 16b DST mode. Works with 32b as well but results in more ops/kernels.
def add_tree_31_to_1(
    in0,
    in1,
    in2,
    in3,
    in4,
    in5,
    in6,
    in7,
    in8,
    in9,
    in10,
    in11,
    in12,
    in13,
    in14,
    in15,
    in16,
    in17,
    in18,
    in19,
    in20,
    in21,
    in22,
    in23,
    in24,
    in25,
    in26,
    in27,
    in28,
    in29,
    in30,
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
    "shape",
    [(64, 64)],  # Single small shape only
    ids=["64x64"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],  # bf16 only
    ids=["bf16"],
)
def test_add_tree_31_to_1_dram(device, shape, dtype):
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=add_tree_31_to_1,
        num_inputs=31,
        buffer_type=ttnn.BufferType.DRAM,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    [((64, 64), (0, 0), ttnn.ShardStrategy.BLOCK)],  # Single small BLOCK config
    ids=["64x64_BLOCK"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
    ids=["bf16"],
)
def test_add_tree_31_to_1_l1(device, shape, max_grid, shard_strategy, dtype):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        add_tree_31_to_1,
        num_inputs=31,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
    )


### ------------------------------------------------------------------------ ###


# A ladder pattern where each add output is summed with the next input sequentially.
# Unlike the balanced tree, this creates a long dependency chain and requires less
# intermediate storage. It's still limited to fusing into 7-input d2m.generics when in 32b mode due to
# the way the CB limit is currently enforced but will be fusable into a single 31x 32b inputs in
# later releases. YMMV based on tensor/grid sizes.
def binary_ladder_31(
    in0,
    in1,
    in2,
    in3,
    in4,
    in5,
    in6,
    in7,
    in8,
    in9,
    in10,
    in11,
    in12,
    in13,
    in14,
    in15,
    in16,
    in17,
    in18,
    in19,
    in20,
    in21,
    in22,
    in23,
    in24,
    in25,
    in26,
    in27,
    in28,
    in29,
    in30,
):
    res_0 = ttnn.add(in0, in1)
    res_1 = ttnn.subtract(res_0, in2)
    res_2 = ttnn.multiply(res_1, in3)
    res_3 = ttnn.subtract(res_2, in4)

    res_4 = ttnn.add(in5, res_3)
    res_5 = ttnn.subtract(in6, res_4)
    res_6 = ttnn.multiply(in7, res_5)
    res_7 = ttnn.subtract(in8, res_6)

    res_8 = ttnn.add(res_7, in9)
    res_9 = ttnn.add(in10, res_8)
    res_10 = ttnn.add(res_9, in11)
    res_11 = ttnn.add(in12, res_10)

    res_12 = ttnn.subtract(res_11, in13)
    res_13 = ttnn.subtract(in14, res_12)
    res_14 = ttnn.subtract(res_13, in15)
    res_15 = ttnn.subtract(in16, res_14)

    res_16 = ttnn.multiply(res_15, in17)
    res_17 = ttnn.multiply(in18, res_16)
    res_18 = ttnn.multiply(res_17, in19)
    res_19 = ttnn.multiply(in20, res_18)

    res_20 = ttnn.subtract(res_19, in21)
    res_21 = ttnn.subtract(in22, res_20)
    res_22 = ttnn.subtract(res_21, in23)
    res_23 = ttnn.subtract(in24, res_22)

    res_24 = ttnn.add(res_23, in25)
    res_25 = ttnn.add(res_24, in26)
    res_26 = ttnn.add(res_25, in27)
    res_27 = ttnn.add(res_26, in28)
    res_28 = ttnn.add(res_27, in29)
    res_29 = ttnn.add(res_28, in30)

    return res_29


@pytest.mark.parametrize(
    "shape",
    [(64, 64)],
    ids=["64x64"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32],  # f32 only as per original test
    ids=["f32"],
)
def test_binary_ladder_31_dram(device, shape, dtype):
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=binary_ladder_31,
        num_inputs=31,
        buffer_type=ttnn.BufferType.DRAM,
    )


@pytest.mark.parametrize(
    "shape, max_grid, shard_strategy",
    [((64, 64), (0, 0), ttnn.ShardStrategy.BLOCK)],
    ids=["64x64_BLOCK"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32],
    ids=["f32"],
)
def test_binary_ladder_31_l1(device, shape, max_grid, shard_strategy, dtype):
    run_op_test(
        device,
        shape,
        max_grid,
        dtype,
        binary_ladder_31,
        num_inputs=31,
        buffer_type=ttnn.BufferType.L1,
        shard_strategy=shard_strategy,
    )


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((2048, 512), torch.bfloat16),
        ((4096, 512), torch.bfloat16),
    ],
    ids=["2048x512_bf16", "4096x512_bf16"],
)
def test_large_shapes_muladd_l1(device, shape, dtype):
    run_op_test(
        device,
        shape,
        max_grid=(7, 7),
        dtype=dtype,
        op=mul_add,
        num_inputs=3,
        buffer_type=ttnn.BufferType.L1,
    )


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((2048, 512), torch.float32),
        ((2048, 512), torch.bfloat16),
        ((4096, 1024), torch.bfloat16),
    ],
    ids=["2048x512_f32", "2048x512_bf16", "4096x1024_bf16"],
)
def test_large_shapes_muladd_dram(device, shape, dtype):
    run_op_test(
        device,
        shape,
        max_grid=(0, 0),
        dtype=dtype,
        op=mul_add,
        num_inputs=3,
        buffer_type=ttnn.BufferType.DRAM,
    )
