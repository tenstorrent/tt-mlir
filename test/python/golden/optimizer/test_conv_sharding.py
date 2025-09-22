# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
import re

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer

pytestmark = pytest.mark.frontend("ttir")


def check_sharded_input_output(mlir_file: str, op_name: str):
    sharded_layouts = []
    with open(mlir_file, "r") as f:
        for line in f:
            if line.startswith("#ttnn_layout") and "sharded" in line:
                layout = line.split("=", 1)[0].strip()
                sharded_layouts.append(layout)

            if len(sharded_layouts) > 0:
                pattern = re.compile(
                    rf".*{op_name}.*({'|'.join(sharded_layouts)}).*->.*({'|'.join(sharded_layouts)}).*"
                )
                if pattern.search(line):
                    return True
    return False


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (16, 32, 32, 64),
            (64, 64, 3, 3),
            (1, 1, 1, 64),
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([1, 1], [1, 1], [1, 1], 1)]
)
def test_conv2d_sharding(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
    device,
):
    def conv2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
            in0,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    output_file_mlir = compile_ttir_to_flatbuffer(
        conv2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=[
            "enable-optimizer=true",
            "memory-layout-analysis-enabled=true",
        ],
        target="ttnn",
    )
    assert check_sharded_input_output(output_file_mlir, "conv2d")
