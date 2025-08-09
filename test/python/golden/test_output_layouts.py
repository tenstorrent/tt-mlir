# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
import re

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer, _is_opmodel_enabled
import os


def check_layouts(mlir_file: str):
    l1 = False
    layout1 = False
    with open(mlir_file, "r") as f:
        for line in f:
            if line.startswith("#l1"):
                l1 = True

            if line.startswith("#ttnn_layout1"):
                layout1 = True
    assert l1, "L1 buffer type not found in the MLIR file"
    assert layout1, "TTNN layout1 not found in the MLIR file"


@pytest.mark.parametrize(
    "shapes",
    [
        (
            32,
            32,
        )
    ],
)
@pytest.mark.parametrize("dtypes", [torch.float32], ids=["f32"])
def test_output_layouts(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
):
    def model(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
    ):
        add_0 = builder.add(in0, in0)
        builder.set_output_layout_override({"buffer_type": "l1"}, add_0)
        return add_0

    output_file_mlir = compile_ttir_to_flatbuffer(
        model,
        [shapes, shapes],
        [dtypes, dtypes],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
    if _is_opmodel_enabled():
        check_layouts(output_file_mlir)
