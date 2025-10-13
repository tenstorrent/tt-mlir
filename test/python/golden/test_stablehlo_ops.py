# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only

from builder.base.builder import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import compile_stablehlo_to_flatbuffer
from test_utils import Marks, shape_str

pytestmark = pytest.mark.frontend("shlo")


def add(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.add(in0, in1, unit_attrs=unit_attrs)


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
    ],
)
def test_binary_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    compile_stablehlo_to_flatbuffer(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


# @pytest.mark.skip("https://github.com/tenstorrent/tt-mlir/issues/5317")
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
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    compile_stablehlo_to_flatbuffer(
        test_fn,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


# Special handling for tan PCC checks. Due to the vertical asymptote on the tan graph, small changes in input values result in large changes in output values at multiples of pi/2, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_tan(shape: Shape, dtype: torch.dtype, target: str, request):
    def tan(
        in0: Operand, builder: StableHLOBuilder, unit_attrs: Optional[List[str]] = None
    ):
        import math

        randn_tensor = torch.randn(shape, dtype=dtype)
        input_golden = randn_tensor.uniform_(
            (-math.pi / 2 + 0.02), (math.pi / 2 - 0.02)
        )
        output_golden = torch.tan(input_golden)
        tan_0 = builder.tan(in0, unit_attrs=unit_attrs)
        builder.set_goldens({in0: input_golden}, {tan_0: output_golden})
        builder.set_graph_level_check(True)
        return tan_0

    compile_stablehlo_to_flatbuffer(
        tan,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
