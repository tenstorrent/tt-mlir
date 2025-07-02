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


@pytest.mark.parametrize("shape", [(32,32),(32,128),(128,32),(256,256)])
def test_dram_read_dma(shape: Shape, request):
    def dram_read_dma(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):

        to_device = builder.to_layout(
            in0,
            output_type=builder.metal_tensor_layout(shape, tilize=False, memorySpace=ttcore.MemorySpace.DeviceDRAM),
            unit_attrs=unit_attrs,
        )

        to_l1 = builder.to_layout(
            to_device,
            output_type=builder.metal_tensor_layout(shape, tilize=False, memorySpace=ttcore.MemorySpace.DeviceL1),
            unit_attrs=unit_attrs,
        )

        from_device = builder.to_layout(
            to_l1,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return from_device

    compile_to_flatbuffer(
        dram_read_dma,
        [shape],
        target="ttmetal",
        custom_pipeline="ttir-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
