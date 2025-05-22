# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder, Shape
from ttmlir.dialects import tt
from ttmlir.ir import *


@pytest.mark.parametrize("shape", [(32, 32)])
def test_to_layout(shape: Shape, request):
    def to_layout(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        to_device = builder.to_layout(
            in0,
            output_type=builder.metal_tensor_layout(shape, (1, 1)),
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
