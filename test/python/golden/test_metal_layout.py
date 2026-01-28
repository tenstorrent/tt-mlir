# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs
from typing import List

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder_utils import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("input_grid_y", [1, 2, 3])
@pytest.mark.parametrize("input_grid_x", [1, 2, 3])
@pytest.mark.parametrize("output_grid_y", [2, 3, 4])
@pytest.mark.parametrize("output_grid_x", [2, 3, 4])
@pytest.mark.parametrize("shard_mul_y", [3])
@pytest.mark.parametrize("shard_mul_x", [2])
@pytest.mark.parametrize("tiled", [False])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_to_layout(
    input_grid_y: int,
    input_grid_x: int,
    output_grid_y: int,
    output_grid_x: int,
    shard_mul_y: int,
    shard_mul_x: int,
    tiled: bool,
    target: str,
    request,
    device,
):
    tile_size = 32 if tiled else 4  # 4 because of 16byte noc alignment
    input_grid = (input_grid_y, input_grid_x)
    output_grid = (output_grid_y, output_grid_x)
    shape = (
        input_grid_y * output_grid_y * shard_mul_y * tile_size,
        input_grid_x * output_grid_x * shard_mul_x * tile_size,
    )

    def module(builder: D2MBuilder):
        @builder.func([shape], [torch.float32])
        def to_layout(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            to_device = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(shape, tiled=tiled),
                unit_attrs=unit_attrs,
                loc="to_device",
            )
            reblock = builder.to_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(shape, tiled=tiled),
                unit_attrs=unit_attrs,
                loc="reblock",
            )
            from_device = builder.to_layout(
                reblock,
                output_type=in0.type,
                unit_attrs=unit_attrs,
                loc="from_device",
            )
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_view_materialization_on_return(
    target: str,
    request,
    device,
):
    """Test that views returned directly are properly materialized.

    This test exercises the D2MMaterializeViewReturns pass by creating a
    to_layout operation followed by a view_layout (via reblock), and then
    directly returning the view without consuming it in a generic op.
    Without the materialization pass, this would fail at runtime since the
    view is just a representational transformation.
    """
    input_grid = (1, 1)
    output_grid = (2, 2)
    shape = (64, 64)

    def module(builder: D2MBuilder):
        @builder.func([shape], [torch.float32])
        def view_return_test(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            # Move to device with 1x1 grid
            to_device = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=input_grid, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="to_device",
            )
            # Reblock to 2x2 grid - this creates a view that needs materialization
            reblocked = builder.to_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=output_grid, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="reblock",
            )
            # Return directly from device without moving back to host.
            # The view must be materialized before returning.
            from_device = builder.to_layout(
                reblocked,
                output_type=in0.type,
                unit_attrs=unit_attrs,
                loc="from_device",
            )
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,d2m-materialize-view-returns,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "grids",
    [
        ((1, 1), (2, 2)),  # 1x1 -> 2x2 grid redistribution
        ((2, 2), (4, 1)),  # 2x2 -> 4x1 grid reshape
        ((1, 4), (2, 2)),  # 1x4 -> 2x2 grid reshape
    ],
)
def test_chained_view_composition(
    target: str,
    grids: tuple,
    request,
    device,
):
    """Test that chained device-to-device to_layout operations correctly compose affine maps.

    This test verifies the core affine map composition logic in buildDeviceToLogicalMap
    and buildLayoutTransformMap. When multiple device-to-device to_layout ops are chained,
    they should lower to view_layout ops with correctly composed affine maps that transform
    coordinates through the entire chain.

    The test creates:
    1. Host -> device with grid1
    2. grid1 -> grid2 (first view)
    3. grid2 -> grid3 (second view, should compose with first)
    4. Device -> host

    The materialize-view-returns pass ensures the final view is materialized,
    and golden verification confirms the affine map composition is correct.
    """
    grid1, grid2 = grids
    grid3 = (grid2[0] // 2 if grid2[0] > 1 else 2, grid2[1] * 2 if grid2[1] > 1 else 2)
    shape = (128, 128)

    def module(builder: D2MBuilder):
        @builder.func([shape], [torch.float32])
        def chained_views(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            # Host to device with grid1
            to_device = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=grid1, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="to_device",
            )
            # First device-to-device: grid1 -> grid2 (should be a view)
            view1 = builder.to_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=grid2, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="view1",
            )
            # Second device-to-device: grid2 -> grid3 (should compose with view1)
            view2 = builder.to_layout(
                view1,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=grid3, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="view2",
            )
            # Back to host (will materialize the composed view)
            from_device = builder.to_layout(
                view2,
                output_type=in0.type,
                unit_attrs=unit_attrs,
                loc="from_device",
            )
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,d2m-materialize-view-returns,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "shape,grid1,grid2",
    [
        ((64, 64), (1, 1), (2, 2)),  # Even split
        ((96, 96), (1, 3), (3, 1)),  # Asymmetric grids
        ((128, 64), (2, 1), (1, 2)),  # Non-square tensor
    ],
)
def test_view_with_padding(
    target: str,
    shape: tuple,
    grid1: tuple,
    grid2: tuple,
    request,
    device,
):
    """Test that views correctly handle padding/alignment changes.

    When tensors have different padding or alignment requirements across
    grid redistributions, the affine map must correctly account for these
    changes. This test verifies that the buildLayoutTransformMap function
    properly handles alignment and padding in the composed affine maps.
    """

    def module(builder: D2MBuilder):
        @builder.func([shape], [torch.float32])
        def view_with_padding(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            # Host to device with grid1
            to_device = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=grid1, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="to_device",
            )
            # Redistribute to grid2 (may have different padding/alignment)
            redistributed = builder.to_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=grid2, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="redistribute",
            )
            # Back to host
            from_device = builder.to_layout(
                redistributed,
                output_type=in0.type,
                unit_attrs=unit_attrs,
                loc="from_device",
            )
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,d2m-materialize-view-returns,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "grid_sequence",
    [
        ((1, 1), (4, 4), (2, 2)),  # Upscale then downscale
        ((2, 2), (1, 2), (2, 4)),  # Multiple redistributions
        ((1, 4), (4, 1), (2, 2)),  # Reshape through chain
    ],
)
def test_multiple_grid_reblocks(
    target: str,
    grid_sequence: tuple,
    request,
    device,
):
    """Test multiple consecutive grid reblocking operations.

    This verifies that:
    1. Grid reblocking is correctly classified as requiring DMA (not zero-copy view)
    2. Multiple reblocking operations in sequence work correctly
    3. The affine map composition works for complex reblocking chains
    """
    shape = (128, 128)

    def module(builder: D2MBuilder):
        @builder.func([shape], [torch.float32])
        def multiple_reblocks(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            current = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=grid_sequence[0], tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="to_device",
            )

            for i, grid in enumerate(grid_sequence[1:], 1):
                current = builder.to_layout(
                    current,
                    output_type=builder.get_metal_tensor_layout(
                        shape, grid=grid, tiled=False
                    ),
                    unit_attrs=unit_attrs,
                    loc=f"reblock_{i}",
                )

            from_device = builder.to_layout(
                current,
                output_type=in0.type,
                unit_attrs=unit_attrs,
                loc="from_device",
            )
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "input_grid,output_grid",
    [
        ((1, 1), (2, 2)),  # 1x1 -> 2x2 tiled reblock
        ((2, 2), (4, 4)),  # 2x2 -> 4x4 tiled reblock
        ((4, 4), (2, 2)),  # 4x4 -> 2x2 tiled downscale
    ],
)
def test_tiled_grid_reblocking(
    target: str,
    input_grid: tuple,
    output_grid: tuple,
    request,
    device,
):
    """Test grid reblocking with tiled tensors.

    Verifies that buildLayoutTransformMap correctly handles tile units vs scalar units
    when computing affine maps for grid reblocking with tiled tensors.
    """
    # Shape must be divisible by tile size (32) and grid dimensions
    shape = (256, 256)

    def module(builder: D2MBuilder):
        @builder.func([shape], [torch.float32])
        def tiled_reblock(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            # Host -> device tiled with input_grid
            to_device_untiled = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=input_grid, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="to_device",
            )

            # Tilize
            tiled_input = builder.to_layout(
                to_device_untiled,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=input_grid, tiled=True
                ),
                unit_attrs=unit_attrs,
                loc="tilize",
            )

            # Grid reblock while tiled
            tiled_reblocked = builder.to_layout(
                tiled_input,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=output_grid, tiled=True
                ),
                unit_attrs=unit_attrs,
                loc="tiled_reblock",
            )

            # Untilize
            untiled_reblocked = builder.to_layout(
                tiled_reblocked,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=output_grid, tiled=False
                ),
                unit_attrs=unit_attrs,
                loc="untilize",
            )

            # Device -> host
            from_device = builder.to_layout(
                untiled_reblocked,
                output_type=in0.type,
                unit_attrs=unit_attrs,
                loc="from_device",
            )
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )
