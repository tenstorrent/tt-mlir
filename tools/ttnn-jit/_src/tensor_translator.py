# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import ttnn, ttcore
from ttnn_jit._src.conversions import (
    ttcore_dtype_from_ttnn_dtype,
    ttcore_dtype_from_mlir_dtype,
    mlir_memory_layout_from_ttnn_memory_layout,
)
import math

DRAM_GRID_SIZE = [1, 1]
OUTPUT_TENSOR_DERIVATION_REQUIRED = ["matmul"]
TILE_WIDTH = 32
TILE_HEIGHT = 32


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


def _calculate_tile_shape(shape):
    """
    Calculate memref shape (tile dimensions) for a given tensor shape.

    For 3D+ tensors, collapses leading dimensions into the first dimension
    before calculating tiles. This matches TTNN's dimension collapsing behavior.

    Args:
        shape: List of tensor dimensions (e.g., [], [64], [64, 64], or [16, 64, 64])

    Returns:
        List representing memref shape:
        - Scalar (len=0): [1, 1]
        - 1D tensor (len=1): [tiles_w] (1D memref)
        - 2D+ tensor (len>=2): [tiles_h, tiles_w] (2D memref)
        Tiles are calculated using ceil division by TILE_WIDTH and TILE_HEIGHT.
    """
    logical_shape = list(shape)

    if len(logical_shape) == 0:
        # Scalar: use 1x1 tile memref
        return [1, 1]
    elif len(logical_shape) == 1:
        # 1D tensor: use 1D memref
        dim_w = logical_shape[0]
        tiles_w = math.ceil(dim_w / TILE_WIDTH)
        return [tiles_w]
    elif len(logical_shape) == 2:
        # 2D tensor: use dimensions directly
        tiles_h = math.ceil(logical_shape[0] / TILE_HEIGHT)
        tiles_w = math.ceil(logical_shape[1] / TILE_WIDTH)
        return [tiles_h, tiles_w]
    else:
        # 3D+ tensor: collapse leading dimensions into the first dimension
        collapsed_shape = [logical_shape[-2], logical_shape[-1]]
        for dim in logical_shape[:-2]:
            collapsed_shape[0] *= dim
        tiles_h = math.ceil(collapsed_shape[0] / TILE_HEIGHT)
        tiles_w = math.ceil(collapsed_shape[1] / TILE_WIDTH)
        return [tiles_h, tiles_w]


def _get_physical_grid_shape(tensor_arg):
    # IMPORTANT: TTNN writes grids as (width, height) but compiler expects (height, width). This function returns (width, height).
    if tensor_arg.memory_config().nd_shard_spec is not None:
        core_range_set = tensor_arg.memory_config().nd_shard_spec.grid
    else:
        core_range_set = tensor_arg.memory_config().shard_spec.grid
    number_of_core_ranges = len(core_range_set.ranges())
    if number_of_core_ranges != 1:
        raise ValueError(
            "Tensor grids with more than one CoreRange are not supported. Tensor grids must be rectangular."
        )

    core_coord = core_range_set.bounding_box().grid_size()
    return (core_coord.x, core_coord.y)


def _create_sharded_tensor_layout(ctx, tensor_arg):
    grid_shape = _get_physical_grid_shape(tensor_arg)
    affine_map = _get_collapsed_linear_affine_map(ctx, tensor_arg.shape, grid_shape)
    buffer_type = ttnn.ir.BufferTypeAttr.get(ctx, ttnn.BufferType.L1)
    # TTNN writes grids as (width, height) but compiler expects (height, width).
    grid = ttcore.ir.GridAttr.get(ctx, list(reversed(grid_shape)))

    shard_spec = tensor_arg.memory_config().shard_spec
    shard_shape = shard_spec.shape
    shard_shape_tile = _calculate_tile_shape(shard_shape)

    data_type = ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
    tile_type = ttcore.ir.TileType.get(ctx, TILE_WIDTH, TILE_HEIGHT, data_type)
    with Location.unknown(ctx):
        memref = MemRefType.get(shard_shape_tile, tile_type, None, buffer_type)

    tensor_mesh = None
    exact_grid = True

    mem_layout = tensor_arg.memory_config().memory_layout.value

    return ttnn.ir.TTNNLayoutAttr.get_with_linear(
        ctx,
        affine_map,
        grid,
        memref,
        mem_layout,
        tensor_mesh,
        exact_grid,
    )


def _create_dram_tensor_layout(ctx, tensor_arg):
    affine_map = _get_collapsed_linear_affine_map(ctx, tensor_arg.shape, DRAM_GRID_SIZE)
    grid = ttcore.ir.GridAttr.get(ctx, DRAM_GRID_SIZE)
    buffer_type = ttnn.ir.BufferTypeAttr.get(ctx, ttnn.BufferType.DRAM)

    data_type = ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
    tile_type = ttcore.ir.TileType.get(ctx, TILE_WIDTH, TILE_HEIGHT, data_type)
    shape = _calculate_tile_shape(tensor_arg.shape)

    with Location.unknown(ctx):
        memref = MemRefType.get(shape, tile_type, None, buffer_type)

    tensor_mesh = None
    exact_grid = True
    return ttnn.ir.TTNNLayoutAttr.get_with_linear(
        ctx,
        affine_map,
        grid,
        memref,
        ttnn.TensorMemoryLayout.Interleaved.value,
        tensor_mesh,
        exact_grid,
    )


def _create_nd_sharded_tensor_layout(context, tensor_arg):
    data_type = ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
    tile_type = ttcore.ir.TileType.get(context, TILE_HEIGHT, TILE_WIDTH, data_type)

    shard_spec = tensor_arg.memory_config().nd_shard_spec
    assert shard_spec is not None, "Expected an ND sharded tensor"

    # TTNN writes grids as (width, height) but compiler expects (height, width)
    grid_shape = _get_physical_grid_shape(tensor_arg)
    grid = ttcore.ir.GridAttr.get(context, list(reversed(grid_shape)))

    # Create memref, tile type only.
    shard_shape = list(shard_spec.shard_shape)
    shard_shape[-2] = shard_shape[-2] // TILE_HEIGHT
    shard_shape[-1] = shard_shape[-1] // TILE_WIDTH
    buffer_type = ttnn.ir.BufferTypeAttr.get(context, ttnn.BufferType.L1)
    memref = MemRefType.get(shard_shape, tile_type, None, buffer_type)

    mem_layout = ttnn.ir.TensorMemoryLayoutAttr.get(
        context,
        mlir_memory_layout_from_ttnn_memory_layout(
            tensor_arg.memory_config().memory_layout
        ),
    )
    shard_orientation = ttnn.ir.ShardOrientationAttr.get(
        context, int(shard_spec.orientation.value)
    )
    shard_distribution_strategy = ttnn.ir.ShardDistributionStrategyAttr.get(
        context, int(shard_spec.shard_distribution_strategy.value)
    )
    ttnn_layout = ttnn.ir.TTNNNDLayoutAttr.get(
        context,
        grid,
        memref,
        mem_layout,
        int(shard_spec.orientation.value),
        int(shard_spec.shard_distribution_strategy.value),
    )
    return ttnn_layout


def _is_dram_layout(layout):
    return (
        ttnn.ir.BufferTypeAttr.maybe_downcast(layout.memref.memory_space).value
        == ttnn.BufferType.DRAM.value
    )


def _check_layout_supported(tensor_arg):

    if tensor_arg.layout.value != ttnn.Layout.Tile:
        raise ValueError(
            f"Only Layout.Tile tensors are supported. Found layout: {tensor_arg.layout}"
        )

    mem_config = tensor_arg.memory_config()
    legacy_sharded = mem_config.shard_spec is not None
    nd_sharded = mem_config.is_sharded() and mem_config.shard_spec is None

    if mem_config.is_sharded():
        if mem_config.buffer_type.value == ttnn.BufferType.DRAM:
            raise ValueError("Sharded DRAM tensors are not supported.")
        if (
            legacy_sharded
            and mem_config.shard_spec.orientation.value
            == ttnn.ShardOrientation.ColMajor
        ):
            raise ValueError("Column major sharding is not supported.")
        if (
            nd_sharded
            and mem_config.nd_shard_spec.orientation.value
            == ttnn.ShardOrientation.ColMajor
        ):
            raise ValueError("Column major sharding is not supported.")
        if (
            nd_sharded
            and mem_config.nd_shard_spec.shard_distribution_strategy.value
            == ttnn.ShardDistributionStrategy.Grid2D
        ):
            raise ValueError("Grid2D distribution strategy is not supported.")

    if mem_config.buffer_type.value == ttnn.BufferType.L1:
        if mem_config.memory_layout == ttnn.TensorMemoryLayout.Interleaved:
            raise ValueError("Interleaved L1 tensors are not supported.")


def _get_output_shape(op_name, input_shapes):
    if op_name == "matmul":
        return [input_shapes[0][0], input_shapes[1][1]]
    else:
        return input_shapes[0]


def _get_virtual_grid_shape(layout):
    if layout.tensor_memory_layout_as_int == ttnn.TensorMemoryLayout.BlockSharded.value:
        return [layout.grid_shape[0], layout.grid_shape[1]]
    elif (
        layout.tensor_memory_layout_as_int
        == ttnn.TensorMemoryLayout.HeightSharded.value
    ):
        return [layout.grid_shape[0] * layout.grid_shape[1], 1]
    elif (
        layout.tensor_memory_layout_as_int == ttnn.TensorMemoryLayout.WidthSharded.value
    ):
        return [1, layout.grid_shape[0] * layout.grid_shape[1]]
    elif (
        layout.tensor_memory_layout_as_int == ttnn.TensorMemoryLayout.Interleaved.value
    ):
        return [1, 1]
    else:
        raise ValueError(
            f"Unsupported memory layout: {layout.tensor_memory_layout_as_int}"
        )


def _infer_block_sharding_grid(shape):
    """Infer a (height, width) grid shape for block sharding the given logical tensor shape"""
    assert len(shape) == 2, f"Only 2D shapes are supported"
    tile_shape = _calculate_tile_shape(shape)
    grid = []
    for dim in tile_shape:
        for grid_dim in reversed(range(1, 9)):
            if dim % grid_dim == 0:
                grid.append(grid_dim)
                break
    return grid


def _get_output_grid_shape(op_name, output_shape, input_layouts):
    if op_name == "matmul":
        in0_grid = _get_virtual_grid_shape(input_layouts[0])
        in1_grid = _get_virtual_grid_shape(input_layouts[1])
        if not _is_dram_layout(input_layouts[0]) and not _is_dram_layout(
            input_layouts[1]
        ):
            return [in0_grid[0], in1_grid[1]]
        else:
            return _infer_block_sharding_grid(output_shape)
    else:
        return _get_virtual_grid_shape(input_layouts[0])


def _create_tensor_layout_with_shape(
    ctx, base_layout, new_shape, new_grid_shape, mem_space, memory_layout
):
    affine_map = _get_collapsed_linear_affine_map(ctx, new_shape, new_grid_shape)

    new_shard_shape = [
        new_shape[0] // new_grid_shape[0] // TILE_HEIGHT,
        new_shape[1] // new_grid_shape[1] // TILE_WIDTH,
    ]
    buffer_type = ttnn.ir.BufferTypeAttr.get(ctx, mem_space)
    with Location.unknown(ctx):
        memref = MemRefType.get(
            new_shard_shape, base_layout.memref.element_type, None, buffer_type
        )
    grid = ttcore.ir.GridAttr.get(ctx, new_grid_shape)

    tensor_mesh = None
    exact_grid = True
    return ttnn.ir.TTNNLayoutAttr.get_with_linear(
        ctx,
        affine_map,
        grid,
        memref,
        memory_layout.value,
        tensor_mesh,
        exact_grid,
    )


def _get_output_memory_space_and_layout(op_name, input_layouts):
    if op_name == "matmul":
        return ttnn.BufferType.L1, ttnn.TensorMemoryLayout.BlockSharded
    else:
        return (
            input_layouts[0].memref.memory_space,
            input_layouts[0].tensor_memory_layout_as_int,
        )


def create_output_tensor(ctx, op_name, input_types, create_encoding=True):
    if op_name not in OUTPUT_TENSOR_DERIVATION_REQUIRED:
        return input_types[0]

    input_layouts = [
        ttnn.ir.TTNNLayoutAttr.maybe_downcast(tensor.encoding) for tensor in input_types
    ]
    shape = _get_output_shape(op_name, [tensor.shape for tensor in input_types])
    grid_shape = _get_output_grid_shape(op_name, shape, input_layouts)
    mem_space, memory_layout = _get_output_memory_space_and_layout(
        op_name, input_layouts
    )
    layout = (
        _create_tensor_layout_with_shape(
            ctx, input_layouts[0], shape, grid_shape, mem_space, memory_layout
        )
        if create_encoding
        else None
    )
    with Location.unknown(ctx):
        output_type = RankedTensorType.get(shape, input_types[0].element_type, layout)

    return output_type


def create_tensor_layout(ctx, tensor_arg):
    """Create TTNN layout attribute from tensor."""
    _check_layout_supported(tensor_arg)

    mem_config = tensor_arg.memory_config()

    if mem_config is not None and mem_config.is_sharded():
        if mem_config.shard_spec is None:
            # Tensor is sharded, but there is no (legacy) shard spec.
            # This means the sharding scheme can only be represented by an ND shard spec.
            return _create_nd_sharded_tensor_layout(ctx, tensor_arg)
        else:
            return _create_sharded_tensor_layout(ctx, tensor_arg)
    else:
        return _create_dram_tensor_layout(ctx, tensor_arg)


def _get_logical_tensor_shape(shape):
    if len(shape) == 0:
        return [1, 1]
    elif len(shape) == 1:
        return [1, shape[0]]
    else:
        return list(shape)


def _get_tile_type(ctx, element_type) -> ttcore.ir.TileType:
    """Create a 32x32 tile type with the appropriate data type for the given element type."""
    data_type = ttcore_dtype_from_mlir_dtype(element_type)
    return ttcore.ir.TileType.get(ctx, TILE_WIDTH, TILE_HEIGHT, data_type)


def create_default_dram_interleaved_layout(
    ctx, shape, element_type
) -> ttnn.ir.TTNNLayoutAttr:
    """
    Create a default DRAM interleaved layout for a given shape.

    This creates a simple, always-valid layout suitable for shape-changing ops
    where the input layout cannot be directly reused. The layout uses:
    - Grid: 1x1 (standard for DRAM interleaved)
    - Buffer type: DRAM (avoids L1 memory pressure)
    - Memory layout: Interleaved (spreads data across DRAM banks)
    - Shard shape: full tensor shape in tiles

    Args:
        ctx: MLIR context
        shape: Output tensor shape (list of ints)
        element_type: MLIR element type (e.g., bf16)

    Returns:
        TTNNLayoutAttr for the given shape
    """
    grid_shape = DRAM_GRID_SIZE  # [1, 1] - standard for interleaved
    memory_layout = ttnn.TensorMemoryLayout.Interleaved.value
    buffer_type = ttnn.ir.BufferTypeAttr.get(ctx, ttnn.BufferType.DRAM)
    tensor_mesh = None
    exact_grid = True

    grid = ttcore.ir.GridAttr.get(ctx, grid_shape)
    logical_shape = _get_logical_tensor_shape(shape)
    affine_map = _get_collapsed_linear_affine_map(ctx, logical_shape, grid_shape)
    shard_shape = _calculate_tile_shape(logical_shape)
    tile_type = _get_tile_type(ctx, element_type)
    with Location.unknown(ctx):
        memref = MemRefType.get(shard_shape, tile_type, None, buffer_type)

    return ttnn.ir.TTNNLayoutAttr.get_with_linear(
        ctx,
        affine_map,
        grid,
        memref,
        memory_layout,
        tensor_mesh,
        exact_grid,
    )
