# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_utils import compile_and_execute_d2m

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
        to_layout,
        [shape],
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
        view_return_test,
        [shape],
        target=target,
        custom_pipeline="d2m-lower-to-layout,d2m-materialize-view-returns,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
