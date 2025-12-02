# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import ttnn, ttcore


def _get_collapsed_linear_affine_map(
    context, shape, grid_shape, collapse_intervals=[(0, -1)]
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
    if len(results) > len(grid_shape):
        results = results[: len(grid_shape)]

    # Pad with leading zeros if the number of results is less than the grid rank
    while len(results) < len(grid_shape):
        results.insert(0, AffineConstantExpr.get(0, context))

    # Simplify affine map by simplifying each expr in results
    for i, expr in enumerate(results):

        expr = AffineExpr.simplify_affine_expr(expr, rank, 0)
        results[i] = expr

    return AffineMap.get(rank, 0, results, context)


def _mlir_memory_layout_from_ttnn_memory_layout(memory_layout):
    """Convert TTNN memory layout to MLIR memory layout enum."""
    print(f"Memory layout: {memory_layout}")
    match str(memory_layout):
        case "TensorMemoryLayout.INTERLEAVED":
            return ttnn.TensorMemoryLayout.Interleaved
        case "TensorMemoryLayout.HEIGHT_SHARDED":
            return ttnn.TensorMemoryLayout.HeightSharded
        case "TensorMemoryLayout.WIDTH_SHARDED":
            return ttnn.TensorMemoryLayout.WidthSharded
        case "TensorMemoryLayout.BLOCK_SHARDED":
            return ttnn.TensorMemoryLayout.BlockSharded
        case _:
            raise ValueError(f"Unsupported memory layout: {memory_layout}")


def _ttcore_dtype_from_ttnn_dtype(dtype):
    """Convert TTNN dtype to TTCore dtype enum."""
    match str(dtype):
        case "DataType.BFLOAT16":
            return ttcore.DataType.BFloat16
        case "DataType.FLOAT32":
            return ttcore.DataType.Float32
        case "DataType.BFLOAT8_B":
            return ttcore.DataType.BFP_BFloat8
        case "DataType.INT32":
            return ttcore.DataType.Int32
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def _get_grid_from_bounding_box(tensor_arg):

    core_range_set = tensor_arg.memory_config().shard_spec.grid
    number_of_core_ranges = len(core_range_set.ranges())
    if number_of_core_ranges != 1:
        raise ValueError(
            f"CoreRangeSet should have only 1 CoreRange, but got {number_of_core_ranges}"
        )

    core_coord = core_range_set.bounding_box().grid_size()
    max_grid = (core_coord.x, core_coord.y)

    return max_grid


def _get_grid(ctx, tensor_arg, memory_layout):
    # IMPORTANT: TTNN writes grids as (width, height) but compiler expects (height, width)

    if memory_layout == ttnn.TensorMemoryLayout.Interleaved:
        return ttcore.ir.GridAttr.get(ctx, [1, 1])

    elif memory_layout in (
        ttnn.TensorMemoryLayout.BlockSharded,
        ttnn.TensorMemoryLayout.HeightSharded,
        ttnn.TensorMemoryLayout.WidthSharded,
    ):

        max_grid = _get_grid_from_bounding_box(tensor_arg)

        grid_size_x = max_grid[0]
        grid_size_y = max_grid[1]

        return ttcore.ir.GridAttr.get(ctx, [grid_size_y, grid_size_x])
    else:
        raise ValueError(f"Invalid memory layout: {memory_layout}")


def _create_sharded_tensor_layout(ctx, tensor_arg):

    max_grid = _get_grid_from_bounding_box(tensor_arg)
    affine_map = _get_collapsed_linear_affine_map(ctx, tensor_arg.shape, max_grid)
    buffer_type = ttnn.ir.BufferTypeAttr.get(ctx, ttnn.BufferType.L1)
    grid = _get_grid(
        ctx,
        tensor_arg,
        _mlir_memory_layout_from_ttnn_memory_layout(
            tensor_arg.memory_config().memory_layout
        ),
    )

    shard_spec = tensor_arg.memory_config().shard_spec
    shard_shape = shard_spec.shape
    shard_shape_tile_x = shard_shape[0] // 32
    shard_shape_tile_y = shard_shape[1] // 32

    data_type = _ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
    tile_type = ttcore.ir.TileType.get(ctx, 32, 32, data_type)
    memref = MemRefType.get(
        [shard_shape_tile_x, shard_shape_tile_y], tile_type, None, buffer_type
    )

    tensor_mesh = None
    exact_grid = True

    return ttnn.ir.TTNNLayoutAttr.get_with_linear(
        ctx,
        affine_map,
        grid,
        memref,
        tensor_arg.memory_config().memory_layout,
        tensor_mesh,
        exact_grid,
    )


def _create_dram_tensor_layout(ctx, tensor_arg):
    affine_map = _get_collapsed_linear_affine_map(ctx, tensor_arg.shape, (1, 1))
    buffer_type = ttnn.ir.BufferTypeAttr.get(ctx, ttnn.BufferType.DRAM)
    grid = _get_grid(ctx, tensor_arg, ttnn.TensorMemoryLayout.Interleaved)

    data_type = _ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
    tile_type = ttcore.ir.TileType.get(ctx, 32, 32, data_type)
    shape = [tensor_arg.shape[0] // 32, tensor_arg.shape[1] // 32]
    memref = MemRefType.get(shape, tile_type, None, buffer_type)

    tensor_mesh = None
    exact_grid = True
    return ttnn.ir.TTNNLayoutAttr.get_with_linear(
        ctx,
        affine_map,
        grid,
        memref,
        ttnn.TensorMemoryLayout.Interleaved,
        tensor_mesh,
        exact_grid,
    )


def _check_layout_supported(tensor_arg):
    # todo: get max grid from tensor arg
    print(f"Checking layout supported for tensor: {tensor_arg}")
    if tensor_arg.memory_config().is_sharded():
        if tensor_arg.memory_config().shard_spec is None:
            raise ValueError(
                "Tensor is sharded but no legacy shard spec is present. ND Sharded tensors are not supported yet."
            )

    if (
        tensor_arg.memory_config().buffer_type == ttnn.BufferType.L1
        and not tensor_arg.memory_config().is_sharded()
    ):
        raise ValueError("Interleaved L1 tensors are not supported.")

    if tensor_arg.memory_config().buffer_type == ttnn.BufferType.DRAM and max_grid != (
        0,
        0,
    ):
        raise ValueError("DRAM tensors must be 1x1 grid.")

    if (
        tensor_arg.memory_config().buffer_type == ttnn.BufferType.DRAM
        and tensor_arg.memory_config().is_sharded()
    ):
        raise ValueError("DRAM tensors must be interleaved.")


def create_tensor_layout(ctx, tensor_arg):
    """Create TTNN layout attribute from tensor."""

    if (
        tensor_arg.memory_config() is not None
        and tensor_arg.memory_config().is_sharded()
    ):
        return _create_sharded_tensor_layout(ctx, tensor_arg)
    else:
        return _create_dram_tensor_layout(ctx, tensor_arg)
