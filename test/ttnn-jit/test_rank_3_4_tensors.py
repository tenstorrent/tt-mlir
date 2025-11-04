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
)

from ttmlir.ir import *

def abs(input_tensor):
    return ttnn.abs(input_tensor)

def collapsed_linear_affine_map(context, shape, grid_shape, collapse_intervals):
  
    rank = len(shape)
    
    # Start with a full identity mapping in a mutable list
    results = [AffineDimExpr.get(i, context) for i in range(rank)]
    print("Initial Results: ", results)

    for begin, end in collapse_intervals:
        # Handle negative indices
        if begin < 0:
            begin += rank
        if end < 0:
            end += rank
        if begin >= end:
            continue

        print(f"Collapsing dimensions from {begin} to {end}...")

        # Build collapsed expression
        collapsed_expr = AffineConstantExpr.get(0, context)
        multiplier = 1
        for d_idx in range(end - 1, begin - 1, -1):

            print(f"  Processing dimension {d_idx} with size {shape[d_idx]} and current multiplier {multiplier}")
            print("pre collapsed_expr:", collapsed_expr)

            dim_expr = AffineDimExpr.get(d_idx, context)
            term = dim_expr * multiplier
            collapsed_expr = term + collapsed_expr
            multiplier *= shape[d_idx]

            print("post collapsed_expr:", collapsed_expr)

        # Replace the range of results with the single collapsed expression
        results = results[:begin] + [collapsed_expr] + results[end:]
        print("Final Results before adjustment: ", results)

    # Truncate results to match the rank of the grid shape
    if len(results) > len(grid_shape):
        results = results[:len(grid_shape)]
    
    print("Results after truncation (if any): ", results)

    # Pad with leading zeros if the number of results is less than the grid rank.
    while len(results) < len(grid_shape):
        results.insert(0, AffineConstantExpr.get(0, context))

    print("Results after padding (if any): ", results)

    #simplify affine map
    for i, expr in enumerate(results):
        #convert expr into affineMap
        print(f"expr before simplification at index {i}: ", expr)
        #simplify expr
        expr = AffineExpr.simplify_affine_expr(expr, rank, 0)
        print(f"expr after simplification at index {i}: ", expr)
        results[i] = expr

    # Create the final map from the constructed results list.
    final_map = AffineMap.get(rank, 0, results, context)

    print("Final Affine Map: ", final_map)
    return final_map

""" def create_rank_n_sharded_tile_tensor(device, shape, max_grid, dtype=torch.float32, int_max=0):
    torch.manual_seed(0)
    if not (dtype.is_floating_point or dtype.is_complex):
        # recreate spatial coverage of fp [0,1] in randn and give some overflow headroom
        high_val = int_max if int_max else torch.iinfo(dtype).max // 2
        torch_tensor = torch.randint(high_val, shape, dtype=dtype)
    else:
        if int_max:
            print("Warning: int_max provided for floating point tensor, ignoring.")
        torch_tensor = torch.randn(shape, dtype=dtype)

    start_coord = ttnn.CoreCoord(0, 0)
    end_coord = ttnn.CoreCoord(max_grid[0], max_grid[1])
    core_range = ttnn.CoreRange(start_coord, end_coord)
    core_range_set = ttnn.CoreRangeSet([core_range])

    # TTNN grids are (Width, Height), while tensor shapes are (Height, Width).
    shard_shape_x = h if max_grid[1] == 0 else h // (max_grid[1] + 1)
    shard_shape_y = w if max_grid[0] == 0 else w // (max_grid[0] + 1)

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

#rank 3 tensor test
@pytest.mark.parametrize("rank_1, rank_2, rank_3", [(4, 5, 6), (2, 3, 4)])
@pytest.mark.parametrize("max_grid", (7, 7))
@pytest.mark.parametrize("op", [abs])
def test_rank_3(device, rank_1, rank_2, rank_3, max_grid, op):
    input_tensor = create_rank_n_sharded_tile_tensor(
        device,
        (rank_1, rank_2, rank_3),
        max_grid,
        dtype=torch.float32,
    )

    # JIT compile the operation
    jit_op = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(op)
    output_tensor = op_jit(input_tensor)

    golden_tensor = op(input_tensor)

    assert memory_configs_equal(output_tensor.memory_config(), golden_tensor.memory_config())
    assert all_close_check(output_tensor, golden_tensor)

 """
@pytest.mark.parametrize("shape", [[2, 3, 64, 128]])
@pytest.mark.parametrize("grid_shape", [[1, 1]])
@pytest.mark.parametrize("collapse_intervals", [[(0, -1)]])
def test_collapsed_affine_map(shape, grid_shape, collapse_intervals):

    context = Context()

    expected_map = AffineMap.get(
        4,
        0,
        [
            ((AffineDimExpr.get(0, context) * 192) + (AffineDimExpr.get(1, context) * 64)) + AffineDimExpr.get(2, context),
            AffineDimExpr.get(3, context)
        ],
        context,
    )

    result_map = collapsed_linear_affine_map(context, shape, grid_shape, collapse_intervals)

    print("Expected Map: ", expected_map)
    print("Result Map: ", result_map)

    assert result_map == expected_map