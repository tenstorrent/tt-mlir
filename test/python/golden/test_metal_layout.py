# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer

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
        builder: TTIRBuilder,
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

    compile_ttir_to_flatbuffer(
        to_layout,
        [shape],
        target=target,
        custom_pipeline="ttir-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
