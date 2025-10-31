# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple

from builder.base.builder import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import compile_and_execute_shlo
from test_utils import shape_str

pytestmark = pytest.mark.frontend("shlo")


def add(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.add(in0, in1, unit_attrs=unit_attrs)


def div(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.div(in0, in1, unit_attrs=unit_attrs)


def max(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.max(in0, in1, unit_attrs=unit_attrs)


def min(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.min(in0, in1, unit_attrs=unit_attrs)


def multiply(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


def subtract(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.subtract(in0, in1, unit_attrs=unit_attrs)


def remainder(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.remainder(in0, in1, unit_attrs=unit_attrs)


def pow(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.pow(in0, in1, unit_attrs=unit_attrs)


def atan2(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.atan2(in0, in1, unit_attrs=unit_attrs)


def shift_right_logical(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.shift_right_logical(in0, in1, unit_attrs=unit_attrs)


def shift_left(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.shift_left(in0, in1, unit_attrs=unit_attrs)


def select(
    condition: Operand,
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.select(condition, in0, in1, unit_attrs=unit_attrs)


def abs(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.abs(in0, unit_attrs=unit_attrs)


def ceil(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.ceil(in0, unit_attrs=unit_attrs)


def cosine(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.cosine(in0, unit_attrs=unit_attrs)


def exp(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.exp(in0, unit_attrs=unit_attrs)


def floor(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.floor(in0, unit_attrs=unit_attrs)


def neg(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.neg(in0, unit_attrs=unit_attrs)


def rsqrt(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.rsqrt(in0, unit_attrs=unit_attrs)


def sine(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.sine(in0, unit_attrs=unit_attrs)


def sqrt(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.sqrt(in0, unit_attrs=unit_attrs)


def logistic(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.logistic(in0, unit_attrs=unit_attrs)


def log(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.log(in0, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        add,
        div,
        max,
        min,
        multiply,
        subtract,
        remainder,
        pow,
        atan2,
        shift_right_logical,
        shift_left,
    ],
)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        abs,
        ceil,
        cosine,
        exp,
        floor,
        log,
        logistic,
        neg,
        rsqrt,
        sine,
        sqrt,
    ],
)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_select_op(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def select_test(
        condition: Operand,
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.select(condition, in0, in1, unit_attrs=unit_attrs)

    # Create a boolean condition tensor shape for select
    compile_and_execute_shlo(
        select_test,
        [shape, shape, shape],  # condition, true_val, false_val
        [torch.bool, dtype, dtype],  # condition is bool, others are the specified dtype
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# Special handling for tan PCC checks. Due to the vertical asymptote on the tan graph, small changes in input values result in large changes in output values at multiples of pi/2, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_tan(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def tan(
        in0: Operand, builder: StableHLOBuilder, unit_attrs: Optional[List[str]] = None
    ):
        import math

        randn_tensor = torch.randn(shape, dtype=dtype)
        input_golden = randn_tensor.uniform_(
            (-math.pi / 2 + 0.05), (math.pi / 2 - 0.05)
        )
        output_golden = torch.tan(input_golden)
        tan_0 = builder.tan(in0, unit_attrs=unit_attrs)
        builder.set_goldens({in0: input_golden}, {tan_0: output_golden})
        builder.set_graph_level_check(True)
        return tan_0

    compile_and_execute_shlo(
        tan,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shapes", [[(64, 64), (64, 64), (64, 64)]], ids=["64x64"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_stablehlo_multi_return_support(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    def multi_return_model(
        in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder
    ):
        builder.set_graph_level_check(True)

        add_result = builder.add(in0, in1)
        exp_result = builder.exp(in2)
        sqrt_result = builder.sqrt(exp_result)

        return exp_result, sqrt_result

    compile_and_execute_shlo(
        multi_return_model,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
