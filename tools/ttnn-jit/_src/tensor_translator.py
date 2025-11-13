# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ttmlir.ir import *
from ttmlir.dialects import ttcore, ttnn


def _ttcore_dtype_from_ttnn_dtype(dtype):
    match int(dtype):
        case 0:
            return ttcore.DataType.BFloat16
        case 1:
            return ttcore.DataType.Float32
        case 7:
            return ttcore.DataType.Int32
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def _mlir_buffer_type_from_ttnn_buffer_type(buffer_type):
    match int(buffer_type):
        case 0:
            return ttnn.BufferType.DRAM
        case 1:
            return ttnn.BufferType.L1
        case _:
            raise ValueError(f"Unsupported buffer type: {buffer_type}")


def _mlir_memory_layout_from_ttnn_memory_layout(memory_layout):
    match int(memory_layout):
        case 0:
            return ttnn.TensorMemoryLayout.Interleaved
        case 2:
            return ttnn.TensorMemoryLayout.HeightSharded
        case 3:
            return ttnn.TensorMemoryLayout.WidthSharded
        case 4:
            return ttnn.TensorMemoryLayout.BlockSharded
        case _:
            raise ValueError(f"Unsupported memory layout: {memory_layout}")


def _mlir_shard_orientation_from_ttnn_shard_orientation(shard_orientation):
    match int(shard_orientation):
        case 0:
            return ttnn.ShardOrientation.RowMajor
        case 1:
            return ttnn.ShardOrientation.ColMajor
        case _:
            raise ValueError(f"Unsupported shard orientation: {shard_orientation}")


def _mlir_shard_distribution_strategy_from_ttnn_shard_distribution_strategy(
    shard_distribution_strategy,
):
    match int(shard_distribution_strategy):
        case 0:
            return ttnn.ShardDistributionStrategy.RoundRobin1D
        case 1:
            return ttnn.ShardDistributionStrategy.Grid2D
        case _:
            raise ValueError(
                f"Unsupported shard distribution strategy: {shard_distribution_strategy}"
            )


def _get_collapsed_linear_affine_map(
    context, shape, grid_rank, collapse_intervals=[(0, -1)]
):
    """
    This function creates an affine map for use in constructing TTNNLayoutAttr.
    Its default behavior must match TTNN's dimension collapsing behavior.
    It is based on collapsedLinearAffineMap() in TTCoreOpsTypes.cpp.
    It collapses tensor dimensions onto an n-dimensional grid, e.g.:

      - 3D tensor onto a 2D grid:
        (d0, d1, d2) -> (d0 <> d1, d2)

      - 4D tensor onto a 2D grid:
        (d0, d1, d2, d3) -> (d0 <> d1 <> d2, d3)

    By default, it collapses the interval [0, -1), which matches TTNN's default
    collapsing. You can specify collapse_intervals for flexible collapsing.

    Examples:

      - 4D tensor onto a 3D grid collapse_intervals=[(1, -1)]:
        (d0, d1, d2, d3) -> (d0, d1 <> d2, d3)

      - 4D tensor onto a 3D grid collapse_intervals=[(0, 2)]:
        (d0, d1, d2, d3) -> (d0 <> d1, d2, d3)

      - 7D tensor onto a 4D grid collapse_intervals=[(0, 3), (-3, -1)]:
        (d0, d1, d2, d3, d4, d5, d6) -> (d0 <> d1 <> d2, d3, d4 <> d5, d6)
    """

    rank = len(shape)

    # Start with a full identity mapping
    results = [AffineDimExpr.get(i, context) for i in range(rank)]

    for interval in collapse_intervals:
        begin, end = interval
        # Handle negative indices
        if begin < 0:
            begin += rank
        if end < 0:
            end += rank
        if begin >= end:
            continue

        # Build collapsed expression
        collapsed_expr = AffineConstantExpr.get(0, context)
        multiplier = 1
        for d_idx in range(end - 1, begin - 1, -1):

            dim_expr = AffineDimExpr.get(d_idx, context)
            term = dim_expr * multiplier
            collapsed_expr = term + collapsed_expr
            multiplier *= shape[d_idx]

        results = results[:begin] + [collapsed_expr] + results[end:]

    # Truncate results to match the rank of the grid shape
    if len(results) > grid_rank:
        results = results[:grid_rank]

    # Pad with leading zeros if the number of results is less than the grid rank
    while len(results) < grid_rank:
        results.insert(0, AffineConstantExpr.get(0, context))

    # Simplify affine map by simplifying each expr in results
    for i, expr in enumerate(results):

        expr = AffineExpr.simplify_affine_expr(expr, rank, 0)
        results[i] = expr

    return AffineMap.get(rank, 0, results, context)


def _create_dram_interleaved_tensor_layout(context, max_grid, tensor_arg):
    data_type = _ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
    tile_type = ttcore.ir.TileType.get(context, 32, 32, data_type)

    affine_map = _get_collapsed_linear_affine_map(
        context, tensor_arg.shape, len(max_grid)
    )
    buffer_type = ttnn.ir.BufferTypeAttr.get(context, ttnn.BufferType.DRAM)
    grid = ttcore.ir.GridAttr.get(context, [1, 1])
    shape = [tensor_arg.shape[0] // 32, tensor_arg.shape[1] // 32]
    memref = MemRefType.get(shape, tile_type, None, buffer_type)
    return ttnn.ir.TTNNLayoutAttr.get_with_linear(
        context,
        affine_map,
        grid,
        memref,
        ttnn.TensorMemoryLayout.Interleaved,
        None,
    )


def _create_block_sharded_tensor_layout(context, max_grid, tensor_arg):
    data_type = _ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
    tile_type = ttcore.ir.TileType.get(context, 32, 32, data_type)

    # Create affine map, should be based of tensor shape
    affine_map = _get_collapsed_linear_affine_map(
        context, tensor_arg.shape, len(max_grid)
    )
    shard_spec = tensor_arg.memory_config().shard_spec
    shard_shape = shard_spec.shape

    # Create ttcore grid atttr based off max_grid passed by user
    # Can't pull grid info from tensor unless it's sharded
    grid_size_x = max_grid[0] + 1
    grid_size_y = max_grid[1] + 1

    # TTNN writes grids as (width, height) but compiler expects (height, width)
    grid = ttcore.ir.GridAttr.get(context, [grid_size_y, grid_size_x])

    # Create memref, tile type only.
    shard_shape_tile_x = shard_shape[0] // 32
    shard_shape_tile_y = shard_shape[1] // 32
    buffer_type = ttnn.ir.BufferTypeAttr.get(context, ttnn.BufferType.L1)
    memref = MemRefType.get(
        [shard_shape_tile_x, shard_shape_tile_y], tile_type, None, buffer_type
    )

    ttnn_layout = ttnn.ir.TTNNLayoutAttr.get_with_linear(
        context,
        affine_map,
        grid,
        memref,
        ttnn.TensorMemoryLayout.BlockSharded,
        None,
    )
    return ttnn_layout


def _create_nd_sharded_tensor_layout(context, max_grid, tensor_arg):
    data_type = _ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
    tile_type = ttcore.ir.TileType.get(context, 32, 32, data_type)

    shard_spec = tensor_arg.memory_config().nd_shard_spec
    assert shard_spec is not None, "Expected an ND sharded tensor"
    shard_shape = list(shard_spec.shard_shape)

    # Create ttcore grid atttr based off max_grid passed by user
    grid_size_x = max_grid[0] + 1
    grid_size_y = max_grid[1] + 1

    # TTNN writes grids as (width, height) but compiler expects (height, width)
    grid = ttcore.ir.GridAttr.get(context, [grid_size_y, grid_size_x])

    # Create memref, tile type only.
    print("shard_shape", shard_shape)
    shard_shape[-2] = shard_shape[-2] // 32
    shard_shape[-1] = shard_shape[-1] // 32
    print("shard_shape", shard_shape)
    buffer_type = ttnn.ir.BufferTypeAttr.get(context, ttnn.BufferType.L1)
    memref = MemRefType.get(shard_shape, tile_type, None, buffer_type)

    mem_layout = ttnn.ir.TensorMemoryLayoutAttr.get(
        context,
        _mlir_memory_layout_from_ttnn_memory_layout(
            tensor_arg.memory_config().memory_layout
        ),
    )  # ND Layouts are denoted as block sharded by default in TTNN, even when not equivalent to legacy block sharding
    shard_orientation = ttnn.ir.ShardOrientationAttr.get(
        context, int(shard_spec.orientation)
    )
    shard_distribution_strategy = ttnn.ir.ShardDistributionStrategyAttr.get(
        context, int(shard_spec.shard_distribution_strategy)
    )
    ttnn_layout = ttnn.ir.TTNNNDLayoutAttr.get(
        context,
        grid,
        memref,
        mem_layout,
        int(shard_spec.orientation),
        int(shard_spec.shard_distribution_strategy),
    )
    return ttnn_layout


def create_tensor_layout(context, max_grid, tensor_arg):
    if tensor_arg.memory_config().is_sharded():
        assert (
            _mlir_buffer_type_from_ttnn_buffer_type(
                tensor_arg.memory_config().buffer_type
            )
            == ttnn.BufferType.L1
        ), "Sharded tensors are only supported in L1"

        if tensor_arg.memory_config().shard_spec is None:
            return _create_nd_sharded_tensor_layout(context, max_grid, tensor_arg)
        else:
            assert (
                _mlir_memory_layout_from_ttnn_memory_layout(
                    tensor_arg.memory_config().memory_layout
                )
                == ttnn.TensorMemoryLayout.BlockSharded
            ), "Sharded tensors are only supported in block sharded"
            return _create_block_sharded_tensor_layout(context, max_grid, tensor_arg)
    else:
        assert (
            _mlir_buffer_type_from_ttnn_buffer_type(
                tensor_arg.memory_config().buffer_type
            )
            == ttnn.BufferType.DRAM
        ), "Interleaved tensors are only supported in DRAM"
        assert (
            max_grid[0] == 0 and max_grid[1] == 0
        ), "The grid for DRAM interleaved tensors is always 1x1"
        return _create_dram_interleaved_tensor_layout(context, max_grid, tensor_arg)
