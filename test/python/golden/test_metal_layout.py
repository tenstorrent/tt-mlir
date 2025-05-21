# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder, Shape
from ttmlir.dialects import tt
from ttmlir.ir import *


def createMetalLayoutTensor(
    ctx,
    shape,
    grid,
    tiled=False,
    memorySpace=tt.MemorySpace.DeviceL1,
    collapseIntervals=[(0, -1)],
    oobVal=tt.OOBVal.Undef,
):
    if isinstance(grid, list) or isinstance(grid, tuple):
        grid = tt.ir.GridAttr.get(ctx, list(grid))
    tensorTy = RankedTensorType.get(shape, F32Type.get(ctx))
    layout = tt.ir.MetalLayoutAttr.get(
        ctx, tensorTy, grid, tiled, memorySpace, collapseIntervals, oobVal
    )
    return RankedTensorType.get(shape, F32Type.get(ctx), layout, Location.unknown(ctx))


@pytest.mark.parametrize("shape", [(32, 32)])
def test_to_layout(shape: Shape, request):
    def to_layout(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        to_device = builder.to_layout(
            in0,
            output_type=createMetalLayoutTensor(builder.get_context(), shape, (1, 1)),
            unit_attrs=unit_attrs,
        )
        from_device = builder.to_layout(
            to_device,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )
        return from_device

    compile_to_flatbuffer(
        to_layout,
        [shape],
        target="ttmetal",
        custom_pipeline="ttir-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
