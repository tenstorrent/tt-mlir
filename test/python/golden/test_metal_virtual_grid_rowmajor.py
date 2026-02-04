# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import sys
from typing import Callable, List, Optional, Tuple, Union
from conftest import x86_only, get_request_kwargs

from ttmlir.dialects import ttir, ttcore
from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def compile_dma_test(
    test_func, shape, request, device, extra_pipeline_options: List[str] = []
):

    # Back to back tolayout ops are normally folded during canonicalization into
    # a single ToLayoutOp representing the final result. The option
    # 'disable-tolayout-folding' prevents this
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


def constructHWShardedAffineMap(
    grid: tuple[int, int], device_grid: tuple[int, int], builder: TTIRBuilder
):

    getConstant = lambda value: AffineConstantExpr.get(value, builder._ctx)
    getDim = lambda index: AffineDimExpr.get(index, builder._ctx)

    assert (
        (grid[0] == 1 or grid[1] == 1) and (grid[0] > 1 or grid[1] > 1),
        "grid must be 1x1 or 1xN or Nx1",
    )

    is_height_sharded = grid[0] > 1
    shard_dim = getDim(0) if is_height_sharded else getDim(1)
    wrap_constant = (
        getConstant(device_grid[0])
        if is_height_sharded
        else getConstant(device_grid[1])
    )

    floordiv_dim = AffineFloorDivExpr.get(shard_dim, wrap_constant)
    mod_dim = AffineModExpr.get(shard_dim, wrap_constant)

    exprs = [
        mod_dim if is_height_sharded else floordiv_dim,
        floordiv_dim if is_height_sharded else mod_dim,
        getDim(2),
        getDim(3),
    ]

    map = AffineMap.get(2 * len(grid), 0, exprs, builder._ctx)
    return map


def constructNDVirtualGridAffineMap(
    grid: List[int], device_grid: tuple[int, int], builder: TTIRBuilder
):

    getConstant = lambda value: AffineConstantExpr.get(value, builder._ctx)
    getDim = lambda index: AffineDimExpr.get(index, builder._ctx)

    floorDiv = lambda expr, constant: AffineFloorDivExpr.get(
        expr, getConstant(constant)
    )
    mod = lambda expr, constant: AffineModExpr.get(expr, getConstant(constant))

    flatIdx = getConstant(0)
    stride = 1
    for (dimIdx, extent) in reversed(list(enumerate(grid))):
        dimStrideExpr = mod(getDim(dimIdx), extent) * stride if extent > 1 else 0
        flatIdx += dimStrideExpr
        stride *= extent

    # reshape flatIdx expr to device grid shape
    grid_exprs = []
    stride = 1
    for extent in reversed(device_grid):
        grid_exprs.append(mod(floorDiv(flatIdx, stride), extent))
        stride *= extent
    grid_exprs = list(reversed(grid_exprs))

    shard_exprs = [getDim(i) for i in range(len(grid), 2 * len(grid))]
    exprs = grid_exprs + shard_exprs

    map = AffineMap.get(2 * len(grid), 0, exprs, builder._ctx)
    return map


@pytest.mark.skip(reason="Still in development")
@pytest.mark.parametrize("target", ["ttmetal"])
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
def test_roundtrip_dma_rowmajor_virtual_grid(
    shape: Shape,
    grid_shape: tuple[int, int],
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

        # derive sharded shapes
        start_shard_shape = [s // g for s, g in zip(shape, grid_shape)]

        start_index_map = (
            constructHWShardedAffineMap(grid_shape, device_grid, builder)
            if len(grid_shape) == 2
            else constructNDVirtualGridAffineMap(grid_shape, device_grid, builder)
        )
        # note: there is a bug in the builder.get_metal_tensor_layout function
        # that returns an incorrect device shape for ND shapes.
        start_output_type = builder.get_metal_tensor_layout(
            shape,
            tiled=False,
            memorySpace=memory_space,
            grid=grid_shape,
            index_map=start_index_map,
        )
        tensor_layoutA = builder.to_layout(
            in0,
            output_type=start_output_type,
            unit_attrs=unit_attrs,
        )

        system_out = builder.to_layout(
            tensor_layoutA,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return system_out

    compile_dma_test(dram_write, shape, request, device=device)
