# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List

from ttir_builder.utils import compile_to_flatbuffer, Marks, shape_str, build_mlir_module
from ttir_builder import Operand, TTIRBuilder, UnitAttr, Shape, TypeInfo
from ttmlir.dialects import ttir, ttcore
from ttmlir.ir import *

@pytest.mark.parametrize("shape", [(x, y) for x in (32, 128) for y in (32, 64, 128)] + [(256,256)])
def test_dram_tilize(shape: Shape, request):
    def tilize(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):

        to_device = builder.to_layout(
            in0,
            output_type=builder.metal_tensor_layout(shape, tiled=False, memorySpace=ttcore.MemorySpace.DeviceDRAM),
            unit_attrs=unit_attrs,
        )

        tilize_out = builder.tilize(
            to_device,
            output_type=builder.metal_tensor_layout(shape, tiled=True, memorySpace=ttcore.MemorySpace.DeviceL1),
            unit_attrs=unit_attrs,
        )

        untilize_out = builder.untilize(
            tilize_out,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return untilize_out

    # Back to back tolayout ops are normally folded during canonicalization into
    # a single ToLayoutOp representing the final result. The option
    # 'disable-tolayout-folding' prevents this
    pipeline_options="{disable-tolayout-folding=1}"
    pipeline = ",".join([
        "ttir-lower-to-layout",
        f"ttir-to-ttmetal-me-pipeline{pipeline_options}",
        f"ttir-to-ttmetal-be-pipeline{pipeline_options}"
    ])
    compile_to_flatbuffer(
        tilize,
        [shape],
        target="ttmetal",
        custom_pipeline=pipeline,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
