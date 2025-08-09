# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
import re

from ttmlir import optimizer_overrides
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer, _is_opmodel_enabled
import os


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
            (1, 1, 1, 64),
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 4])
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
):
    def conv2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        conv2d_0 = builder.conv2d(
            in0,
            weight,
            bias,
            in1,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )
        builder.set_conv2d_config_override()
        return conv2d_0

    output_file_mlir = compile_ttir_to_flatbuffer(
        conv2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
    if _is_opmodel_enabled():
        assert check_sharded_input_output(output_file_mlir, "conv2d")


def check_policy(mlir_file: str):
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


@pytest.mark.subprocess
@pytest.mark.parametrize(
    "shapes",
    [(1, 32, 32, 64)],
)
@pytest.mark.parametrize("dtypes", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "optimization_policy",
    [optimizer_overrides.MemoryLayoutAnalysisPolicyType.BFInterleaved],
)
def test_optimization_policies(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    optimization_policy: optimizer_overrides.MemoryLayoutAnalysisPolicyType,
    request,
):
    def model(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
    ):
        return builder.add(in0, in0)

    output_file_mlir = compile_ttir_to_flatbuffer(
        model,
        [shapes, shapes],
        [dtypes, dtypes],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        optimization_policy=optimization_policy,
    )
    if _is_opmodel_enabled():
        check_policy(output_file_mlir)


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


@pytest.mark.subprocess
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
