# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for tile masking functionality.

These tests verify that the d2m-decompose-masking pass correctly handles
out-of-bounds masking for tensors that require padding for tile alignment.

The tests use explicit dim_alignments to create tensors where some tiles
are completely outside the logical bounds, and verify the padding regions
are filled with the correct OOBVal.
"""

import pytest
import torch
from typing import List
from conftest import get_request_kwargs

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m

pytestmark = pytest.mark.frontend("ttir")


def make_golden_with_padding(
    input_tensor: torch.Tensor,
    aligned_shape: tuple,
    fill_value: float,
) -> torch.Tensor:
    """Create golden tensor with input in upper-left and fill_value elsewhere."""
    golden = torch.full(aligned_shape, fill_value, dtype=input_tensor.dtype)
    input_shape = input_tensor.shape
    # Copy input to upper-left corner
    golden[: input_shape[0], : input_shape[1]] = input_tensor
    return golden


@pytest.mark.parametrize(
    "logical_shape,aligned_shape,fill_value,oobval",
    [
        # 32x32 logical with 64x64 alignment creates 2x2 tile grid where
        # tiles (0,1), (1,0), (1,1) are completely outside logical bounds
        ((32, 32), (64, 64), 0.0, ttcore.OOBVal.Zero),
        ((32, 32), (64, 64), 1.0, ttcore.OOBVal.One),
        ((32, 32), (64, 64), float("-inf"), ttcore.OOBVal.NegInf),
        ((32, 32), (64, 64), float("inf"), ttcore.OOBVal.Inf),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_complete_tile_masking(
    logical_shape: Shape,
    aligned_shape: tuple,
    fill_value: float,
    oobval: ttcore.OOBVal,
    target: str,
    request,
    device,
):
    """Test complete tile OOB masking with various fill values.

    By setting dim_alignments larger than logical_shape, we force extra
    tile padding. Tiles whose starting position is >= logical_shape will
    be completely filled with the OOBVal.

    The output tensor has the aligned_shape, with:
    - Upper-left logical_shape region containing the input data
    - Remaining regions filled with fill_value (the OOBVal)
    """
    # Create input with known values
    input_tensor = torch.ones(logical_shape, dtype=torch.float32) * 42.0

    # Golden: input in upper-left, fill_value elsewhere
    golden = make_golden_with_padding(input_tensor, aligned_shape, fill_value)

    def module(builder: D2MBuilder):
        # Input is logical_shape, output is aligned_shape
        @builder.func([logical_shape], [torch.float32])
        def tilize_with_complete_tile_mask(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            # Set custom input and expected golden output
            builder.set_goldens(inputs={in0: input_tensor})

            # Tilize with explicit dim_alignments larger than logical shape
            # This creates complete tiles that are entirely OOB
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    logical_shape,
                    tiled=True,
                    oobVal=oobval,
                    dim_alignments=aligned_shape,
                ),
                unit_attrs=unit_attrs,
            )

            # View as tiled with the ALIGNED logical shape
            # Keep tiled=True so the final to_layout will properly untilize
            view_with_aligned_logical = builder.view_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    aligned_shape, tiled=True, oobVal=oobval
                ),
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )

            # Output type is the aligned shape - to_layout will untilize
            output_type = RankedTensorType.get(aligned_shape, F32Type.get(builder._ctx))
            from_device = builder.to_layout(
                view_with_aligned_logical,
                output_type=output_type,
                unit_attrs=unit_attrs,
            )

            # Set expected output golden
            builder.set_goldens(inputs={}, outputs={from_device: golden})

            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        # d2m-decompose-masking is now part of ttir-to-ttmetal-me-pipeline (after bufferization)
        # ttcore-mark-functions-as-forward is needed for flatbuffer translation
        custom_pipeline="ttcore-mark-functions-as-forward,d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
        print_ir="/tmp/ir_dumps",  # DEBUG: dump IR after each pass to this directory
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_tilize_no_masking_when_aligned(shape: Shape, target: str, request, device):
    """Test that tile-aligned tensors without explicit dim_alignments work correctly.

    When tensors are already tile-aligned (multiples of 32) and we don't force
    extra alignment, the masking pass should essentially be a no-op (no OOB tiles).
    """
    # Create deterministic input
    input_tensor = torch.randn(shape, dtype=torch.float32)

    def module(builder: D2MBuilder):
        @builder.func([shape], [torch.float32])
        def tilize_aligned(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            # Set input and golden - output should match input (roundtrip)
            builder.set_goldens(inputs={in0: input_tensor}, outputs={})

            # No explicit dim_alignments - uses natural tile alignment
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=True, oobVal=ttcore.OOBVal.Zero
                ),
                unit_attrs=unit_attrs,
            )

            # For aligned shapes, we can directly untilize via to_layout
            # No need for view_layout since logical shape equals aligned shape
            from_device = builder.to_layout(
                to_device,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            # Golden is the input - roundtrip should preserve data
            builder.set_goldens(inputs={}, outputs={from_device: input_tensor})

            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        # d2m-decompose-masking is now part of ttir-to-ttmetal-me-pipeline (after bufferization)
        # ttcore-mark-functions-as-forward is needed for flatbuffer translation
        custom_pipeline="ttcore-mark-functions-as-forward,d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize(
    "logical_shape,aligned_shape,grid_shape,fill_value,oobval",
    [
        # 64x64 logical with 128x128 alignment on explicit 4x4 grid
        # Each core gets 32x32 (1 tile), masking fills OOB regions
        # - Cores (0,0), (0,1), (1,0), (1,1): full data
        # - Cores (2,*), (*,2), etc: partial or complete OOB
        # Use fill_value=1.0 (not 0.0) to catch bugs where masking doesn't run
        # properly - we'd see 42.0 in masked regions instead of 1.0
        ((64, 64), (128, 128), (4, 4), 1.0, ttcore.OOBVal.One),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_multicore_tile_masking(
    logical_shape: Shape,
    aligned_shape: tuple,
    grid_shape: tuple,
    fill_value: float,
    oobval: ttcore.OOBVal,
    target: str,
    request,
    device,
):
    """Test tile masking with explicit multicore grid.

    This test specifies an explicit grid shape in the layout, so the tensor
    starts as 4x4x1x1 (multicore from the beginning). Each core handles a
    shard and must correctly mask OOB regions based on logical bounds.
    """
    # Create input with known values
    input_tensor = torch.ones(logical_shape, dtype=torch.float32) * 42.0

    # Golden: input in upper-left, fill_value elsewhere
    golden = make_golden_with_padding(input_tensor, aligned_shape, fill_value)

    def module(builder: D2MBuilder):
        @builder.func([logical_shape], [torch.float32])
        def tilize_multicore_mask(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            builder.set_goldens(inputs={in0: input_tensor})

            # Tilize to device with explicit grid - creates 4x4x1x1 tensor
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    logical_shape,
                    tiled=True,
                    oobVal=oobval,
                    dim_alignments=aligned_shape,
                    grid=grid_shape,  # Explicit 4x4 grid
                ),
                unit_attrs=unit_attrs,
            )

            # Reinterpret to aligned logical shape (stays on same grid)
            view_with_aligned_logical = builder.view_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    aligned_shape, tiled=True, oobVal=oobval, grid=grid_shape
                ),
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )

            output_type = RankedTensorType.get(aligned_shape, F32Type.get(builder._ctx))
            from_device = builder.to_layout(
                view_with_aligned_logical,
                output_type=output_type,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens(inputs={}, outputs={from_device: golden})
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        # Use custom pipeline (skips GridSelection since we set grid explicitly)
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize(
    "logical_shape,aligned_shape,fill_value,oobval",
    [
        # Partial tile cases: logical shape not aligned to tile boundaries
        # 50x50 logical with 64x64 alignment: tile (0,0) has partial data
        # - Rows 0-49 are valid, rows 50-63 should be filled
        # - Cols 0-49 are valid, cols 50-63 should be filled
        ((50, 50), (64, 64), 0.0, ttcore.OOBVal.Zero),
        ((50, 50), (64, 64), 1.0, ttcore.OOBVal.One),
        # 50x64: partial row masking only (cols are fully aligned)
        ((50, 64), (64, 64), float("inf"), ttcore.OOBVal.Inf),
        # 64x50: partial col masking only (rows are fully aligned)
        ((64, 50), (64, 64), float("-inf"), ttcore.OOBVal.NegInf),
        # 18x18: very small logical shape, most of first tile is OOB
        ((18, 18), (32, 32), 0.0, ttcore.OOBVal.Zero),
        # 100x100: spans multiple tiles, last tile is partial
        ((100, 100), (128, 128), 0.0, ttcore.OOBVal.Zero),
        ((50, 50), (128, 128), 1.0, ttcore.OOBVal.One),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_partial_tile_masking(
    logical_shape: Shape,
    aligned_shape: tuple,
    fill_value: float,
    oobval: ttcore.OOBVal,
    target: str,
    request,
    device,
):
    """Test partial tile OOB masking with various fill values.

    When logical_shape is not aligned to tile boundaries (32x32), some tiles
    will straddle the boundary. Elements within the logical bounds should
    preserve input data, while elements outside should be filled with OOBVal.

    For example, with logical_shape=(50, 50) and aligned_shape=(64, 64):
    - Tile (0,0): rows 0-49 and cols 0-49 contain data, rest is filled
    - Tiles (0,1), (1,0), (1,1): completely OOB, entirely filled
    """
    # Create input with known values
    input_tensor = torch.ones(logical_shape, dtype=torch.float32) * 42.0

    # Golden: input in upper-left logical_shape region, fill_value elsewhere
    golden = make_golden_with_padding(input_tensor, aligned_shape, fill_value)

    def module(builder: D2MBuilder):
        @builder.func([logical_shape], [torch.float32])
        def tilize_with_partial_tile_mask(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            builder.set_goldens(inputs={in0: input_tensor})

            # Tilize with explicit dim_alignments larger than logical shape
            # This creates partial tiles that straddle the boundary
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    logical_shape,
                    tiled=True,
                    oobVal=oobval,
                    dim_alignments=aligned_shape,
                ),
                unit_attrs=unit_attrs,
            )

            # View as tiled with the ALIGNED logical shape
            view_with_aligned_logical = builder.view_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    aligned_shape, tiled=True, oobVal=oobval
                ),
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )

            # Output type is the aligned shape - to_layout will untilize
            output_type = RankedTensorType.get(aligned_shape, F32Type.get(builder._ctx))
            from_device = builder.to_layout(
                view_with_aligned_logical,
                output_type=output_type,
                unit_attrs=unit_attrs,
            )

            # Set expected output golden
            builder.set_goldens(inputs={}, outputs={from_device: golden})

            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        # d2m-decompose-masking is now part of ttir-to-ttmetal-me-pipeline (after bufferization)
        # ttcore-mark-functions-as-forward is needed for flatbuffer translation
        custom_pipeline="ttcore-mark-functions-as-forward,d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        save_artifacts=True,
    )


@pytest.mark.parametrize(
    "logical_shape,aligned_shape,grid_shape,fill_value,oobval",
    [
        ((100, 100), (128, 128), (4, 4), 1.0, ttcore.OOBVal.One),
        ((60, 100), (128, 128), (4, 4), 0.0, ttcore.OOBVal.Zero),
        ((100, 60), (128, 128), (4, 4), 1.0, ttcore.OOBVal.One),
        ((60, 60), (128, 128), (4, 4), 1.0, ttcore.OOBVal.One),
        ((100, 100), (128, 128), (2, 2), 1.0, ttcore.OOBVal.One),
        ((60, 60), (128, 128), (2, 2), 1.0, ttcore.OOBVal.One),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_multicore_partial_tile_masking(
    logical_shape: Shape,
    aligned_shape: tuple,
    grid_shape: tuple,
    fill_value: float,
    oobval: ttcore.OOBVal,
    target: str,
    request,
    device,
):
    """Test partial tile masking with explicit multicore grid.

    This test verifies that partial tile masking works correctly in a multicore
    scenario where multiple cores each handle a shard, and some cores have
    tiles that straddle the logical boundary.
    """
    # Create input with known values
    input_tensor = torch.ones(logical_shape, dtype=torch.float32) * 42.0

    # Golden: input in upper-left, fill_value elsewhere
    golden = make_golden_with_padding(input_tensor, aligned_shape, fill_value)

    def module(builder: D2MBuilder):
        @builder.func([logical_shape], [torch.float32])
        def tilize_multicore_partial_mask(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            builder.set_goldens(inputs={in0: input_tensor})

            # Tilize to device with explicit grid - creates 4x4x1x1 tensor
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    logical_shape,
                    tiled=True,
                    oobVal=oobval,
                    dim_alignments=aligned_shape,
                    grid=grid_shape,  # Explicit 4x4 grid
                ),
                unit_attrs=unit_attrs,
            )

            # Reinterpret to aligned logical shape (stays on same grid)
            view_with_aligned_logical = builder.view_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    aligned_shape, tiled=True, oobVal=oobval, grid=grid_shape
                ),
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )

            output_type = RankedTensorType.get(aligned_shape, F32Type.get(builder._ctx))
            from_device = builder.to_layout(
                view_with_aligned_logical,
                output_type=output_type,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens(inputs={}, outputs={from_device: golden})
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        # Use custom pipeline (skips GridSelection since we set grid explicitly)
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
