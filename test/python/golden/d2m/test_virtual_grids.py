# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional

from conftest import get_request_kwargs

from ttmlir.dialects import ttcore
from ttmlir.ir import (
    AffineConstantExpr,
    AffineDimExpr,
    AffineFloorDivExpr,
    AffineMap,
    AffineModExpr,
)
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")


def create_tileid_debug_tensor(shape: Shape, dtype: torch.dtype):
    """Create a debug tensor where each value in a 32x32 tile equals its multi-dimensional tile ID.

    The tile grid is computed over the last two dimensions (32x32 tiles); outer dimensions get unique
    tile coordinates but tile extent is only 32x32 in the inner dimensions. The scalar tile ID uses
    row-major ordering across the full multi-dimensional tile grid.
    """
    TILE_DIM_SIZE = int(32)
    assert len(shape) >= 2, "Shape must have at least 2 dimensions."
    shape = tuple(int(d) for d in shape)
    ndim = len(shape)

    # Compute number of tiles along each dimension (ceil division for partial tiles)
    num_tiles_per_dim = [(dim + TILE_DIM_SIZE - 1) // TILE_DIM_SIZE for dim in shape]

    # Generate meshgrids for tile coordinates across all dimensions
    tile_coords_list = []
    for i in range(ndim):
        dim_indices = torch.arange(shape[i])
        # For outer dims (i < ndim-2), use full dimension indexing as "tile coords"
        # For inner two dims (i >= ndim-2), use 32-sized tile indexing
        if i >= ndim - 2:
            tile_coord = dim_indices // TILE_DIM_SIZE
        else:
            tile_coord = (
                dim_indices  # Each element in outer dim is its own "tile coord"
            )
        tile_coords_list.append(tile_coord)

    # Compute the meshgrid of tile coordinates across all dimensions
    tile_coords = torch.meshgrid(*tile_coords_list, indexing="ij")

    # Compute the scalar tile ID using row-major ordering across the full grid
    tile_id = torch.zeros_like(tile_coords[0])
    stride = 1
    for i in reversed(range(ndim)):
        tile_id += tile_coords[i] * stride
        stride *= num_tiles_per_dim[i]

    return tile_id.to(dtype)


def compile_rowmajor_dma_test(test_func, shape, request, device):

    # Back to back to_layout ops are normally folded during canonicalization
    # into a single ToLayoutOp representing the final result. The
    # disable-tolayout-folding option preserves both transitions.
    pipeline_options = "{disable-tolayout-folding=1 collapse-tensors-2d=0}"
    pipeline = ",".join(
        [
            f"ttir-to-ttmetal-pipeline{pipeline_options}",
        ]
    )
    compile_and_execute_ttir(
        test_func,
        [shape],
        target="ttmetal",
        device=device,
        custom_pipeline=pipeline,
        **get_request_kwargs(request),
    )


def construct_hw_sharded_affine_map(
    grid: tuple[int, ...], device_grid: tuple[int, int], builder: TTIRBuilder
):

    get_constant = lambda value: AffineConstantExpr.get(value, builder._ctx)
    get_dim = lambda index: AffineDimExpr.get(index, builder._ctx)

    assert (grid[0] == 1 or grid[1] == 1) and (
        grid[0] > 1 or grid[1] > 1
    ), "grid must be 1x1 or 1xN or Nx1"

    is_height_sharded = grid[0] > 1
    shard_dim = get_dim(0) if is_height_sharded else get_dim(1)
    wrap_constant = (
        get_constant(device_grid[0])
        if is_height_sharded
        else get_constant(device_grid[1])
    )

    floordiv_dim = AffineFloorDivExpr.get(shard_dim, wrap_constant)
    mod_dim = AffineModExpr.get(shard_dim, wrap_constant)

    exprs = [
        mod_dim if is_height_sharded else floordiv_dim,
        floordiv_dim if is_height_sharded else mod_dim,
        get_dim(2),
        get_dim(3),
    ]

    return AffineMap.get(2 * len(grid), 0, exprs, builder._ctx)


def construct_nd_virtual_grid_affine_map(
    grid: tuple[int, ...], device_grid: tuple[int, int], builder: TTIRBuilder
):

    get_constant = lambda value: AffineConstantExpr.get(value, builder._ctx)
    get_dim = lambda index: AffineDimExpr.get(index, builder._ctx)

    floor_div = lambda expr, constant: AffineFloorDivExpr.get(
        expr, get_constant(constant)
    )
    mod = lambda expr, constant: AffineModExpr.get(expr, get_constant(constant))

    flat_idx = get_constant(0)
    stride = 1
    for dim_idx, extent in reversed(list(enumerate(grid))):
        dim_stride_expr = mod(get_dim(dim_idx), extent) * stride if extent > 1 else 0
        flat_idx += dim_stride_expr
        stride *= extent

    grid_exprs = []
    stride = 1
    for extent in reversed(device_grid):
        grid_exprs.append(mod(floor_div(flat_idx, stride), extent))
        stride *= extent
    grid_exprs = list(reversed(grid_exprs))

    shard_exprs = [get_dim(i) for i in range(len(grid), 2 * len(grid))]
    exprs = grid_exprs + shard_exprs

    return AffineMap.get(2 * len(grid), 0, exprs, builder._ctx)


@pytest.mark.parametrize(
    "shape",
    [
        (32, 4096),
        (4096, 32),
        (2048, 32),
        (32, 1280),  # uses 1x40 grid
        (1536, 64),  # uses 48x1 grid
        (1120, 32),  # uses 35x1 grid
        (32, 768),  # uses 1x24 grid
        (1, 1, 1, 1, 128, 128),
        (1, 1, 1, 1, 2, 32, 512),
        (1, 1, 1, 1, 32, 32),
        (1, 1, 1, 4, 128, 256),
        (1, 1, 2, 1, 1, 512, 64),
        (1, 1, 32, 128),
        (1, 1, 32, 32),
        (1, 1, 64, 32),
        (1, 2, 1, 1, 1, 128, 32),
        (1, 2, 1, 1, 2, 1024, 32),
        (1, 2, 1, 4, 128, 32),
        (1, 2, 1, 4, 32, 64),
        (1, 2, 256, 128),
        (1, 4, 2, 1, 128, 32),
        (1, 512, 32),
        (2, 1, 1, 1, 1, 256, 256),
        (2, 1, 2, 1, 512, 32),
        (2, 1, 2, 256, 32),
        (2, 1, 256, 256),
        (2, 1, 4, 64, 64),
        (2, 2, 1, 2, 128, 64),
        (2, 32, 64),
        (2, 4, 4, 1, 2, 1, 32, 32),
        (32, 32, 128),
        (32, 32, 32),
        (4, 1, 2, 2, 1, 64, 128),
        (4, 2, 1, 2, 1, 64, 32),
        (4, 2, 32, 512),
        (4, 256, 128),
        (4, 32, 32),
        (4, 64, 64),
        (8, 1, 512, 32),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_virtual_grid_eltwise(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def eltwise_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            # input_tensor = create_tileid_debug_tensor(shape, dtype)
            input_tensor = torch.randint(0, 1001, shape, dtype=dtype)

            # abs is an identity function for positive integers, so use it for debugging ease
            result = builder.abs(in0, unit_attrs=unit_attrs)

            golden_output_tensor = torch.abs(input_tensor).to(dtype)
            builder.set_goldens({in0: input_tensor}, {result: golden_output_tensor})

            return result

    # device shape override is needed so that shapes are equivalently divisible
    # on both WH and BH.
    options = [f"collapse-tensors-2d=0", "override-device-shape=8,8"]

    compile_and_execute_ttir(
        module,
        device=device,
        **get_request_kwargs(request),
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}} ",
        target=target,
    )


@pytest.mark.skip(reason="Still in development")
@pytest.mark.parametrize(
    "shape, grid_shape",
    [
        ((32, 2048), (1, 64)),
        ((2048, 32), (64, 1)),
        ((1, 32, 2048), (1, 1, 64)),
        ((1, 512, 512), (1, 8, 8)),
        ((1, 256, 512), (1, 8, 8)),
        ((1, 512, 256), (1, 8, 8)),
        ((1, 512, 512), (1, 4, 4)),
        ((1, 256, 512), (1, 4, 4)),
        ((1, 512, 256), (1, 4, 4)),
        ((1, 512, 512), (1, 2, 4)),
        ((1, 256, 512), (1, 2, 4)),
        ((1, 512, 256), (1, 2, 4)),
        ((1, 512, 512), (1, 4, 2)),
        ((1, 256, 512), (1, 4, 2)),
        ((1, 512, 256), (1, 4, 2)),
        ((4, 512, 512), (4, 4, 2)),
        ((4, 256, 512), (4, 4, 2)),
        ((4, 512, 256), (4, 4, 2)),
        ((8, 64, 512), (4, 1, 8)),
        ((8, 64, 512), (4, 1, 8)),
        ((8, 64, 256), (4, 1, 2)),
        ((1, 8, 64, 512), (1, 4, 1, 8)),
        ((1, 8, 64, 512), (1, 4, 1, 8)),
        ((1, 8, 64, 256), (1, 4, 1, 2)),
        ((8, 1, 64, 512), (1, 1, 2, 8)),
        ((8, 1, 64, 512), (1, 1, 2, 8)),
        ((8, 1, 64, 256), (1, 1, 2, 2)),
    ],
)
@pytest.mark.parametrize("memory_space", [ttcore.MemorySpace.DeviceL1])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_roundtrip_dma_rowmajor_virtual_grid(
    shape: Shape,
    grid_shape: tuple[int, ...],
    memory_space: ttcore.MemorySpace,
    target: str,
    request,
    device,
):
    def dram_write(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):

        device_grid = (8, 8)

        start_index_map = (
            construct_hw_sharded_affine_map(grid_shape, device_grid, builder)
            if len(grid_shape) == 2
            else construct_nd_virtual_grid_affine_map(grid_shape, device_grid, builder)
        )
        # There is a bug in builder.get_metal_tensor_layout that returns an
        # incorrect device shape for ND shapes.
        start_output_type = builder.get_metal_tensor_layout(
            shape,
            tiled=False,
            memorySpace=memory_space,
            grid=grid_shape,
            index_map=start_index_map,
        )
        tensor_layout = builder.to_layout(
            in0,
            output_type=start_output_type,
            unit_attrs=unit_attrs,
        )

        system_out = builder.to_layout(
            tensor_layout,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return system_out

    compile_rowmajor_dma_test(dram_write, shape, request, device=device)
