# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import torch
from typing import List

from ttmlir.dialects import ttir
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir
from golden import get_golden_function

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("shapes", [[(32, 32), (32, 32), (32, 32)]], ids=["32x32"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
def test_golden_report_accuracy(
    shapes: List[Shape], dtypes: List[torch.dtype], request, device
):
    def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        add = builder.add(in0, in0)
        subtract = builder.subtract(add, in2)
        mult = builder.multiply(in1, subtract)

        # set incorrect input golden for intermediate output tensor of add operation
        zeros_tensor = torch.zeros(shapes[0], dtype=dtypes[0])
        builder.set_goldens(
            inputs={}, outputs={add: zeros_tensor}, set_all_outputs=False
        )
        return mult

    mlir_path = compile_and_execute_ttir(
        model,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        export_golden_report=True,
    )

    # Check golden report exists and stores the correct number of ops
    report_path = mlir_path + ".golden_report.json"
    assert os.path.exists(report_path), f"Golden report not found at {report_path}"
    with open(report_path, "r") as f:
        report = json.load(f)
    assert isinstance(report, dict) and len(report) == 3

    # Ensure ttir.add failed golden comparison
    for entry in report.values():
        if entry.get("op_name") == "ttir.add":
            assert (
                entry.get("result") == "fail"
            ), "ttir.add should fail golden comparison"
        if entry.get("op_name") == "ttir.subtract":
            assert (
                entry.get("result") == "pass"
            ), "ttir.subtract should pass golden comparison"


@pytest.mark.parametrize("shape", [(1, 31, 31, 32)], ids=["1x31x31x32"])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode,count_include_pad",
    [
        ([3, 3], [1, 1], [1, 1], [1, 1, 1, 1], False, False),
    ],
)
def test_get_original_op_loc(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    request,
    device,
):
    def avg_pool2d(
        in0: Operand,
        builder: TTIRBuilder,
    ):
        return builder.avg_pool2d(
            in0,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    mlir_path = compile_and_execute_ttir(
        avg_pool2d,
        [shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        export_golden_report=True,
    )

    # Check golden report exists and stores the correct number of ops
    report_path = mlir_path + ".golden_report.json"
    assert os.path.exists(report_path), f"Golden report not found at {report_path}"
    with open(report_path, "r") as f:
        report = json.load(f)
    assert isinstance(report, dict) and len(report) == 1
