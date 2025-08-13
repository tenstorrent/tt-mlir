# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from conftest import x86_only

from builder.base.builder import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from test_utils import Marks, shape_str


def exp(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.exp(in0, unit_attrs=unit_attrs)


def expm1(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.expm1(in0, unit_attrs=unit_attrs)


def ceil(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.ceil(in0, unit_attrs=unit_attrs)


def floor(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.floor(in0, unit_attrs=unit_attrs)


def abs(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.abs(in0, unit_attrs=unit_attrs)


def logical_not(
    in0: Operand,
    builder: TTIRBuilder,
    shape: Shape,
    dtype: torch.dtype,
    unit_attrs: Optional[List[str]] = None,
):
    randn_tensor = torch.randn(shape, dtype=torch.float32)
    input_tensor = randn_tensor.uniform_(-10.0, 10.0)
    input_tensor[torch.abs(input_tensor) < 4.0] = 0.0
    input_tensor = input_tensor.to(dtype)
    # Torch returns bool tensor but ttnn doesn't have bool type, convert to input dtype.
    golden_output_tensor = torch.logical_not(input_tensor).to(dtype)
    builder.set_graph_input_output(
        [input_tensor], [golden_output_tensor], override=True
    )
    return builder.logical_not(in0, unit_attrs=unit_attrs)


# TODO (wenbinlyuTT): test int32 once untilize issue is fixed
@pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4079")
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_logical_not(shape: Shape, dtype: torch.dtype, target: str, request):
    def logical_not_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return logical_not(in0, builder, shape, dtype, unit_attrs)

    compile_ttir_to_flatbuffer(
        logical_not_wrapper,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_logical_not(shape: Shape, dtype: torch.dtype, target: str, request):
    def hoisted_logical_not_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return logical_not(in0, builder, shape, dtype, unit_attrs=["ttir.should_hoist"])

    compile_ttir_to_flatbuffer(
        hoisted_logical_not_wrapper,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


def bitwise_not(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.bitwise_not(in0, unit_attrs=unit_attrs)


def neg(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.neg(in0, unit_attrs=unit_attrs)


def sign(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sign(in0, unit_attrs=unit_attrs)


def sin(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sin(in0, unit_attrs=unit_attrs)


def cos(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cos(in0, unit_attrs=unit_attrs)


# Special handling for tan PCC checks. Due to the vertical asymptote on the tan graph, small changes in input values result in large changes in output values at multiples of pi/2, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_tan(shape: Shape, dtype: torch.dtype, target: str, request):
    def tan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        import math

        randn_tensor = torch.randn(shape, dtype=dtype)
        input_golden = randn_tensor.uniform_(
            (-math.pi / 2 + 0.02), (math.pi / 2 - 0.02)
        )
        output_golden = torch.tan(input_golden)
        builder.set_graph_input_output([input_golden], [output_golden], override=True)
        return builder.tan(in0, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        tan,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


def atan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.atan(in0, unit_attrs=unit_attrs)


def tanh(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.tanh(in0, unit_attrs=unit_attrs)


# Special handling for log PCC checks. Due to the vertical asymptote on the log graph, small changes in input values result in large changes in output values at negative values, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_log(shape: Shape, dtype: torch.dtype, target: str, request):
    def log(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        randn_tensor = torch.randn(shape, dtype=dtype)
        abs_tensor = torch.abs(randn_tensor)
        error_margin = torch.full(randn_tensor.shape, 0.01)
        input_golden = torch.add(abs_tensor, error_margin)
        output_golden = torch.log(input_golden)
        builder.set_graph_input_output([input_golden], [output_golden], override=True)
        return builder.log(in0, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        log,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


# Special handling for log1p PCC checks. Due to the vertical asymptote on the log1p graph, small changes in input values result in large changes in output values at values below -1, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_log1p(shape: Shape, dtype: torch.dtype, request):
    def log1p(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        randn_tensor = torch.randn(shape, dtype=dtype)
        abs_tensor = torch.abs(randn_tensor)
        error_margin = torch.full(randn_tensor.shape, -0.99)
        input_golden = torch.add(abs_tensor, error_margin)
        output_golden = torch.log1p(input_golden)
        builder.set_graph_input_output([input_golden], [output_golden], override=True)
        return builder.log1p(in0, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        log1p,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def relu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.relu(in0, unit_attrs=unit_attrs)


def gelu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.gelu(in0, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("max_arg,min_arg", [(3.0, 2.0)])
def test_clamp_scalar(shape: Shape, max_arg: float, min_arg: float, request):
    def clamp_scalar(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.clamp_scalar(
            in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
        )

    compile_ttir_to_flatbuffer(
        clamp_scalar,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(32, 64), (32, 64), (32, 64), (32, 64)]])
def test_clamp_tensor(shapes: List[Shape], request):
    def clamp_tensor(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        in3: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.clamp_tensor(in0, in1, in2, in3, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        clamp_tensor,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def leaky_relu(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.leaky_relu(in0, unit_attrs=unit_attrs)


@pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4080")
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_sqrt(shape: Shape, dtype: torch.dtype, target: str, request):
    def sqrt(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        input_tensor = torch.abs(torch.randn(shape, dtype=dtype))
        golden_output_tensor = torch.sqrt(input_tensor)
        builder.set_graph_input_output(
            [input_tensor], [golden_output_tensor], override=True
        )
        return builder.sqrt(in0, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        sqrt,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


def cbrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cbrt(in0, unit_attrs=unit_attrs)


@pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4081")
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_rsqrt(shape: Shape, dtype: torch.dtype, target: str, request):
    def rsqrt(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        input_tensor = torch.abs(torch.randn(shape, dtype=dtype))
        golden_output_tensor = torch.rsqrt(input_tensor)
        builder.set_graph_input_output(
            [input_tensor], [golden_output_tensor], override=True
        )
        return builder.rsqrt(in0, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        rsqrt,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


def sigmoid(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sigmoid(in0, unit_attrs=unit_attrs)


def reciprocal(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.reciprocal(in0, unit_attrs=unit_attrs)


def is_finite(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.is_finite(in0, unit_attrs=unit_attrs)


def get_dimension_size(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.get_dimension_size(in0, unit_attrs=unit_attrs)


@pytest.mark.parametrize(
    "shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
    [
        (
            [(4, 10, 3, 5, 7), (4, 10, 5, 7, 3), (4, 10, 3, 7, 10, 7, 3)],
            [0],
            [3],
            [0],
            [2],
        )
    ],
)
def test_dot_general(
    shapes: List[Shape],
    batch_dims_lhs: List[int],
    contract_dims_lhs: List[int],
    batch_dims_rhs: List[int],
    contract_dims_rhs: List[int],
    request,
):
    def dot_general(
        in0: Operand,
        in1: Operand,
        out0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.dot_general(
            in0,
            in1,
            out0,
            batch_dims_lhs,
            contract_dims_lhs,
            batch_dims_rhs,
            contract_dims_rhs,
            unit_attrs=unit_attrs,
        )

    compile_ttir_to_flatbuffer(
        dot_general,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def add(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.add(in0, in1, unit_attrs=unit_attrs)


def multiply(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


def logical_and(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_and(in0, in1, unit_attrs=unit_attrs)


def logical_or(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_or(in0, in1, unit_attrs=unit_attrs)


def logical_xor(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_xor(in0, in1, unit_attrs=unit_attrs)


def bitwise_and(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_and(in0, in1, unit_attrs=unit_attrs)


def bitwise_or(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_or(in0, in1, unit_attrs=unit_attrs)


def bitwise_xor(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_xor(in0, in1, unit_attrs=unit_attrs)


def subtract(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.subtract(in0, in1, unit_attrs=unit_attrs)


def eq(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.eq(in0, in1, unit_attrs=unit_attrs)


def ne(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.ne(in0, in1, unit_attrs=unit_attrs)


def ge(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.ge(in0, in1, unit_attrs=unit_attrs)


def gt(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.gt(in0, in1, unit_attrs=unit_attrs)


def le(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.le(in0, in1, unit_attrs=unit_attrs)


def lt(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.lt(in0, in1, unit_attrs=unit_attrs)


def div(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    dividend_tensor = builder._get_golden_tensor(in0)
    divisor_tensor = builder._get_golden_tensor(in1)
    if torch.is_floating_point(dividend_tensor) and torch.is_floating_point(
        divisor_tensor
    ):
        dividend_tensor[torch.abs(dividend_tensor) < 0.01] = 0.03
        divisor_tensor[torch.abs(divisor_tensor) < 0.01] = -0.03
    output_golden = torch.div(dividend_tensor, divisor_tensor)
    builder.set_graph_input_output(
        [dividend_tensor, divisor_tensor], [output_golden], override=True
    )
    return builder.div(in0, in1, unit_attrs=unit_attrs)


# TODO (wenbinlyuTT): fix f32 accuracy issue for small values
@pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4082")
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_div(shape: Shape, dtype: torch.dtype, target: str, request):
    compile_ttir_to_flatbuffer(
        div,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_div(shape: Shape, dtype: torch.dtype, target: str, request):
    def hoisted_div_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return div(in0, in1, builder, unit_attrs=["ttir.should_hoist"])

    compile_ttir_to_flatbuffer(
        hoisted_div_wrapper,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


def remainder(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.remainder(in0, in1, unit_attrs=unit_attrs)


def maximum(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.maximum(in0, in1, unit_attrs=unit_attrs)


def minimum(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.minimum(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shapes", [[(10, 64, 32), (32, 128), (128,)]])
def test_linear(shapes: List[Shape], request):
    def linear(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.linear(in0, in1, in2, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        linear,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def pow(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    randn_base_tensor = builder._get_golden_tensor(in0)
    randn_exponent_tensor = builder._get_golden_tensor(in1)
    if torch.is_floating_point(randn_exponent_tensor):
        randn_base_tensor = torch.abs(randn_base_tensor)
    output_golden = torch.pow(randn_base_tensor, randn_exponent_tensor)
    builder.set_graph_input_output(
        [randn_base_tensor, randn_exponent_tensor], [output_golden], override=True
    )
    return builder.pow(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_pow(shape: Shape, dtype: torch.dtype, target: str, request):
    compile_ttir_to_flatbuffer(
        pow,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@x86_only
@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_pow(shape: Shape, dtype: torch.dtype, target: str, request):
    def hoisted_pow_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return pow(in0, in1, builder, unit_attrs=["ttir.should_hoist"])

    compile_ttir_to_flatbuffer(
        hoisted_pow_wrapper,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


def matmul(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.matmul(in0, in1, unit_attrs=unit_attrs)


def sum(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sum(in0, unit_attrs=unit_attrs)


def mean(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.mean(in0, unit_attrs=unit_attrs)


def max(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.max(in0, unit_attrs=unit_attrs)


def min(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.min(in0, unit_attrs=unit_attrs)


def reshape(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    # Calculate total elements in the input tensor
    input_shape = builder.get_shape(in0)
    total_elements = 1
    for dim in input_shape:
        total_elements *= dim

    # Reshape to a 1D tensor with all elements
    new_shape = [int(total_elements)]  # This must be a list of integers
    return builder.reshape(in0, new_shape, unit_attrs=unit_attrs)


def transpose(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.transpose(in0, unit_attrs=unit_attrs)


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dim_arg", [0])
@pytest.mark.parametrize("keep_dim", [False])
def test_prod(shape: Shape, dim_arg: int, keep_dim: bool, request):
    def prod(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.prod(in0, [dim_arg], keep_dim, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        prod,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def where(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.where(in0, in1, in2, unit_attrs=unit_attrs)


def broadcast(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    broadcast_dimensions: Optional[List[int]] = None,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.broadcast(
        in0, in1, broadcast_dimensions=broadcast_dimensions, unit_attrs=unit_attrs
    )


def concat(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    dim: int,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.concat([in0, in1, in2], dim=dim, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shapes", [[(1, 1, 32), (1, 16, 32)]])
@pytest.mark.parametrize("broadcast_dimensions", [[1, 16, 1]])
def test_broadcast(shapes: List[Shape], broadcast_dimensions: List[int], request):
    # Create a wrapper function that captures broadcast_dimensions
    def broadcast_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return broadcast(in0, in1, builder, broadcast_dimensions, unit_attrs)

    # Set the name for better test identification
    broadcast_wrapper.__name__ = "broadcast"

    compile_ttir_to_flatbuffer(
        broadcast_wrapper,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(1, 128, 128, 1)])
@pytest.mark.parametrize("dim", [0])
def test_squeeze(shape: Shape, dim: int, request):
    def squeeze(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.squeeze(in0, dim, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        squeeze,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dim", [0])
def test_unsqueeze(shape: Shape, dim: int, request):
    def unsqueeze(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.unsqueeze(in0, dim, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        unsqueeze,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(1, 32, 32)])
@pytest.mark.parametrize("dims", [[32, 1, 1]])
def test_repeat(shape: Shape, dims: List[int], request):
    def repeat(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.repeat(in0, dims=dims, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        repeat,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 8, 1, 12, 64),
            (1, 8, 1, 12, 64),
        ]
    ],
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("repeats", [1])
def test_repeat_interleave(shapes: List[Shape], repeats: int, dim: int, request):
    def repeat_interleave(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.repeat_interleave(
            in0, in1, repeats=repeats, dim=dim, unit_attrs=unit_attrs
        )

    compile_ttir_to_flatbuffer(
        repeat_interleave,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (32, 128),
            (16, 128),
        ]
    ],
)
@pytest.mark.parametrize("dim", [0])
def test_concat(shapes: List[Shape], dim: int, request):
    # Create a wrapper function that captures dim
    def concat_wrapper(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return concat(in0, in1, in2, dim, builder, unit_attrs)

    # Set the name for better test identification.
    concat_wrapper.__name__ = "concat"

    compile_ttir_to_flatbuffer(
        concat_wrapper,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 32, 3, 3),
            (1, 1, 1, 64),
            (1, 16, 28, 64),
        ]
    ],
)
@pytest.mark.parametrize(
    "input_dtypes",
    [
        [torch.float32, torch.float32, torch.float32, torch.float32],
        # skip quint8 for now. Issue: https://github.com/tenstorrent/tt-metal/issues/26568
        pytest.param(
            [
                TypeInfo(torch.quint8, scale=0.1, zero_point=128),
                TypeInfo(torch.qint8, scale=0.1, zero_point=0),
                torch.float32,
                torch.int8,
            ],
            marks=pytest.mark.skip(
                reason="Issue: https://github.com/tenstorrent/tt-metal/issues/26568"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([2, 1], [2, 1], [2, 1], 2)]
)
def test_conv2d(
    shapes: List[Shape],
    input_dtypes: List[Union[torch.dtype, TypeInfo]],
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
        return builder.conv2d(
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

    compile_ttir_to_flatbuffer(
        conv2d,
        shapes,
        input_dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 32, 3, 3),
            (1, 1, 1, 64),
            (1, 16, 28, 64),
        ]
    ],
)
@pytest.mark.parametrize("stride", [[2, 1]])
@pytest.mark.parametrize("dilation", [[2, 1]])
@pytest.mark.parametrize("padding", [[2, 1]])
@pytest.mark.parametrize("groups", [2])
def test_conv2d_consteval(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
):
    def conv2d_consteval(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
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

    compile_ttir_to_flatbuffer(
        conv2d_consteval,
        shapes,
        argument_types_string="conv2d_consteval=input,parameter,parameter,parameter",
        test_base=request.node.name,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (3, 8, 8, 256),
            (256, 256, 3, 3),
            (1, 1, 1, 256),
            (1, 10, 10, 256),
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 4])
@pytest.mark.parametrize(
    "stride,padding,output_padding,dilation,groups", [(1, 0, 0, 1, 1)]
)
def test_conv_transpose2d(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    groups: int,
    request,
):
    def conv_transpose2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv_transpose2d(
            in0,
            weight,
            bias,
            in1,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    compile_ttir_to_flatbuffer(
        conv_transpose2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode",
    [([2, 2], [2, 2], [1, 1], [0, 0, 0, 0], False)],
)
@pytest.mark.parametrize("shapes", [[(1, 128, 128, 32), (1, 64, 64, 32)]])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 2])
def test_max_pool2d(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    request,
):
    def max_pool2d(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.max_pool2d(
            in0,
            in1,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            unit_attrs=unit_attrs,
        )

    compile_ttir_to_flatbuffer(
        max_pool2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 64, 32, 32),  # input tensor: (N, C, H, W)
            (64,),  # scale (gamma)
            (64,),  # offset (beta)
            (64,),  # mean
            (64,),  # variance
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 5])
@pytest.mark.parametrize("dimension", [1])  # channel dimension
@pytest.mark.parametrize("epsilon", [1e-5])
@pytest.mark.parametrize("training", [False])
def test_batch_norm(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    dimension: int,
    epsilon: float,
    training: bool,
    request,
):
    def batch_norm(
        in0: Operand,
        scale: Operand,
        offset: Operand,
        mean: Operand,
        variance: Operand,
        builder,
        unit_attrs: Optional[List[str]] = None,
    ):

        return builder.batch_norm(
            in0,
            scale,
            offset,
            mean,
            variance,
            epsilon=epsilon,
            dimension=dimension,
            training=training,
            unit_attrs=unit_attrs,
        )

    compile_ttir_to_flatbuffer(
        batch_norm,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("shapes", [[(1, 1, 5, 5), (2, 6, 14, 18)]])
@pytest.mark.parametrize("padding", [[0, 1, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("value", [0])
def test_pad(shapes: List[Shape], padding: List[int], value: int, request):
    def pad(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.pad(
            in0, in1, padding=padding, value=value, unit_attrs=unit_attrs
        )

    compile_ttir_to_flatbuffer(
        pad,
        inputs_shapes=shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(32, 64)])
@pytest.mark.parametrize("dim,begin,end,step", [(0, 0, 3, 1)])
def test_index(shape: Shape, dim: int, begin: int, end: int, step: int, request):
    def index(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.index(
            in0, dim=dim, begin=begin, end=end, step=step, unit_attrs=unit_attrs
        )

    compile_ttir_to_flatbuffer(
        index,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim,begin,length,stride", [(1, 2, 2, 2)])
def test_select(shape: Shape, dim: int, begin: int, length: int, stride: int, request):
    def select(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.select(
            in0,
            dim=dim,
            begin=begin,
            length=length,
            stride=stride,
            unit_attrs=unit_attrs,
        )

    compile_ttir_to_flatbuffer(
        select,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# TODO (ctod): These three nullary tensor creation ops can probably be combined in some way.
@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int32], ids=["bf16", "f32", "i32"]
)
def test_zeros(shape: Shape, dtype: torch.dtype, request):
    def zeros(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.zeros(shape, dtype, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        zeros,
        inputs_shapes=[],
        inputs_types=[],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
def test_ones(shape: Shape, request):
    def ones(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.ones(shape, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        ones,
        inputs_shapes=[],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(128, 128)]])
@pytest.mark.parametrize("dim_arg", [[1]])
def test_argmax(shapes, dim_arg, request):
    def argmax(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.argmax(in0, dim_arg, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        argmax,
        inputs_shapes=shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip("`reverse` doesn't have a legalization. See issue #2495")
@pytest.mark.parametrize("shape", [(64, 64)])
@pytest.mark.parametrize("dims", [[0, 1]])
def test_reverse(shape: Shape, dims: List[int], request):
    def reverse(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reverse(in0, dims=dims, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        reverse,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip("See issue #3685")
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_and(shape: Shape, dim_args: List[int], request):
    def reduce_and(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reduce_and(in0, dim_args=dim_args, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        reduce_and,
        [shape],
        [torch.int32],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def reduce_or(
    in0: Operand,
    builder: TTIRBuilder,
    dim_args: List[int],
    keep_dim: bool = False,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_or(
        in0, dim_args=dim_args, keep_dim=keep_dim, unit_attrs=unit_attrs
    )


@pytest.mark.skip(
    "Generated flatbuffer will currently fail to run due to only floats being supported by the runtime. See issue #1775"
)
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_or(shape: Shape, dim_args: List[int], request):
    def reduce_or_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return reduce_or(in0, builder, dim_args=dim_args, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        reduce_or_wrapper,
        [shape],
        [torch.int32],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def permute(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    permutation: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.permute(
        in0,
        in1,
        permutation=permutation,
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize("shapes", [[(2, 3, 4), (3, 4, 2)]])
@pytest.mark.parametrize("permutation", [[1, 2, 0]])
def test_permute(shapes: List[Shape], permutation: List[int], request):
    # Create a wrapper function that captures permutation
    def permute_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return permute(in0, in1, builder, permutation, unit_attrs)

    # Set the name for better test identification
    permute_wrapper.__name__ = "permute"

    compile_ttir_to_flatbuffer(
        permute_wrapper,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(10, 64, 32, 3), (10, 128, 128, 3)]])
@pytest.mark.parametrize("scale_factor", [[2, 4]])
def test_upsample2d(shapes: List[Shape], scale_factor: List[int], request):
    def upsample2d(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.upsample2d(
            in0,
            in1,
            scale_factor=scale_factor,
            unit_attrs=unit_attrs,
        )

    compile_ttir_to_flatbuffer(
        upsample2d,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape,start,end,step,dim", [((5,), 0, 5, 1, 0)])
def test_arange(shape: Shape, start: int, end: int, step: int, dim: int, request):
    def arange(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.arange(in0, start, end, step, dim, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        arange,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize(
    "from_type,to_type", [(torch.int32, torch.float32)], ids=["i32-f32"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_typecast(
    shape: Shape, from_type: torch.dtype, to_type: torch.dtype, target: str, request
):
    def typecast(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.typecast(in0, in1, unit_attrs=unit_attrs)

    pipeline_options = []
    # Workaround for ttmetal, only support 1x1 grid atm
    if target == "ttmetal":
        pipeline_options.append("override-device-shape=1,1")
    compile_ttir_to_flatbuffer(
        typecast,
        [shape, shape],
        [from_type, to_type],
        test_base=request.node.name,
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize("shapes", [[(4, 4, 128, 128), (4, 4, 128, 128)]])
@pytest.mark.parametrize("dim", [1])
def test_cumsum(shapes: List[Shape], dim: int, request):
    def cumsum(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.cumsum(in0, in1, dim=dim, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        cumsum,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def prod(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.prod(in0, [1], False, unit_attrs=unit_attrs)


@pytest.mark.fails_golden
@pytest.mark.parametrize("shapes", [[(1, 32, 64, 512), (1, 32, 3, 512)]])
def test_fill_cache(shapes: List[Shape], request):
    def fill_cache(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.fill_cache(in0, in1, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        fill_cache,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def softmax(
    in0: Operand,
    builder: TTIRBuilder,
    dimension: int = -1,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.softmax(in0, dimension=dimension, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(512, 1024)])
@pytest.mark.parametrize("dimension", [-1])
def test_softmax(shape: Shape, dimension: int, request):
    # Create a wrapper function that captures dimension
    def softmax_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return softmax(in0, builder, dimension, unit_attrs)

    # Set the name for better test identification
    softmax_wrapper.__name__ = "softmax"

    compile_ttir_to_flatbuffer(
        softmax_wrapper,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.run_error
@pytest.mark.parametrize("shapes", [[(1, 32, 64, 512), (1, 32, 1, 512), (1,)]])
@pytest.mark.parametrize("dtypes", [[torch.float32, torch.float32, torch.int32]])
def test_update_cache(shapes: List[Shape], dtypes: List[torch.dtype], request):
    def update_cache(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.update_cache(in0, in1, in2, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        update_cache,
        shapes,
        inputs_types=dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def embedding(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.embedding(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.qint32,
        pytest.param(
            torch.qint8,
            marks=pytest.mark.skip(
                reason="qint8 quantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
    ids=["qint32", "qint8"],
)
def test_quantize(
    shape: Shape, scale: float, zero_point: int, dtype: torch.dtype, request
):
    def quantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.quantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        quantize,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize(
    "input_dtype",
    [
        TypeInfo(torch.qint32, 0.1, 0),
        pytest.param(
            TypeInfo(torch.qint8, 0.1, 0),
            marks=pytest.mark.skip(
                reason="qint8 dequantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
    ids=["qint32", "qint8"],
)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_dequantize(
    shape: Shape,
    input_dtype: TypeInfo,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
):
    def dequantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.dequantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        dequantize,
        [shape],
        inputs_types=[input_dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize(
    "input_dtype",
    [
        TypeInfo(torch.qint32, 0.1, 0),
        pytest.param(
            TypeInfo(torch.qint8, 0.1, 0),
            marks=pytest.mark.skip(
                reason="qint8 requantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.qint32,
        pytest.param(
            torch.qint8, marks=pytest.mark.skip(reason="qint8 quantize not supported")
        ),
    ],
    ids=["qint32", "qint8"],
)
def test_requantize(
    shape: Shape,
    input_dtype: TypeInfo,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
):
    def requantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.requantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        requantize,
        [shape],
        inputs_types=[input_dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Create hoisted versions of operations by currying the unit_attrs parameter
def create_hoisted_unary_op(op_func, name):
    """Create a hoisted version of a unary operation by adding the should_hoist unit attribute"""

    def hoisted_op(in0, builder, **kwargs):
        # For unary ops
        return op_func(in0, builder, unit_attrs=["ttir.should_hoist"], **kwargs)

    # Set the name for better test identification
    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_binary_op(op_func, name):
    """Create a hoisted version of a binary operation by adding the should_hoist unit attribute"""

    def hoisted_op(in0, in1, builder, **kwargs):
        return op_func(in0, in1, builder, unit_attrs=["ttir.should_hoist"], **kwargs)

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_permute_op(op_func, name):
    """Create a hoisted version of the permute operation that calculates appropriate permutation dimensions"""

    def hoisted_op(in0, in1, builder, **kwargs):
        # Calculate appropriate permutation based on input dimensions
        input_shape = builder.get_shape(in0)
        ndims = len(input_shape)

        # Create a simple permutation that reverses the dimensions
        # This is guaranteed to be valid for any tensor
        permutation = list(range(ndims))
        permutation.reverse()

        return op_func(
            in0, in1, builder, permutation, unit_attrs=["ttir.should_hoist"], **kwargs
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_softmax_op(op_func, name):
    """Create a hoisted version of the softmax operation"""

    def hoisted_op(in0, builder, **kwargs):
        # Default dimension for the hoisted version (last dimension)
        default_dimension = -1
        return op_func(
            in0,
            builder,
            dimension=default_dimension,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_concat_op(op_func, name):
    """Create a hoisted version of the concat operation"""

    def hoisted_op(in0, in1, in2, builder, **kwargs):
        # Default dimension for the hoisted version (dimension 0)
        default_dim = 0
        return op_func(
            in0,
            in1,
            in2,
            default_dim,
            builder,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create a function for hoisted where operation
def create_hoisted_where_op(op_func, name):
    """Create a hoisted version of the where operation"""

    def hoisted_op(condition, x, y, builder, **kwargs):
        return op_func(
            condition, x, y, builder, unit_attrs=["ttir.should_hoist"], **kwargs
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create a function for hoisted slice operation
def create_hoisted_slice_op(op_func, name):
    """Create a hoisted version of the slice operation"""

    def hoisted_op(in0, builder, **kwargs):
        # Default slice parameters
        begins = DenseI32ArrayAttr.get([0, 0])
        ends = DenseI32ArrayAttr.get([10, 10])
        steps = DenseI32ArrayAttr.get([1, 1])
        return op_func(
            in0,
            begins,
            ends,
            steps,
            builder,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create a function for hoisted reduce operations
def create_hoisted_reduce_op(op_func, name):
    """Create a hoisted version of a reduce operation that requires dimension arguments"""

    def hoisted_op(in0, builder, **kwargs):
        # Default dimension arguments for the hoisted version
        default_dim_args = [0]  # Use first dimension as default
        return op_func(
            in0,
            builder,
            dim_args=default_dim_args,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create hoisted versions of all hoistable operations with proper names
hoisted_unary_ops = [
    create_hoisted_unary_op(exp, "exp"),
    create_hoisted_unary_op(abs, "abs"),
    create_hoisted_unary_op(ceil, "ceil"),
    create_hoisted_unary_op(floor, "floor"),
    create_hoisted_unary_op(tanh, "tanh"),
    create_hoisted_unary_op(reciprocal, "reciprocal"),
    create_hoisted_unary_op(neg, "neg"),
    create_hoisted_unary_op(sigmoid, "sigmoid"),
    create_hoisted_unary_op(sin, "sin"),
    create_hoisted_unary_op(cos, "cos"),
    create_hoisted_unary_op(sum, "sum"),
    create_hoisted_unary_op(relu, "relu"),
    pytest.param(
        create_hoisted_unary_op(softmax, "softmax"),
        marks=pytest.mark.xfail(
            reason="Softmax does not lower to loops properly https://github.com/tenstorrent/tt-mlir/issues/3232"
        ),
    ),
    create_hoisted_unary_op(reshape, "reshape"),
    create_hoisted_unary_op(transpose, "transpose"),
]


hoisted_binary_ops = [
    create_hoisted_binary_op(add, "add"),
    create_hoisted_binary_op(multiply, "multiply"),
    create_hoisted_binary_op(subtract, "subtract"),
    create_hoisted_binary_op(eq, "equal"),
    create_hoisted_binary_op(ne, "not_equal"),
    create_hoisted_binary_op(gt, "greater_than"),
    create_hoisted_binary_op(ge, "greater_equal"),
    create_hoisted_binary_op(lt, "less_than"),
    create_hoisted_binary_op(le, "less_equal"),
]


hoisted_ternary_ops = [
    create_hoisted_concat_op(concat, "concat"),
]


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("test_fn", hoisted_unary_ops)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_unary_ops(
    test_fn: Callable,
    shape: Shape,
    request,
    target: str,
    dtype: torch.dtype = torch.float32,
):
    """Test unary ops that support CPU hoisting"""
    compile_ttir_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=f"{request.node.name}",
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        [(128, 128), (128, 128)],  # Same shapes
        [(128, 128), (1, 128)],  # Broadcasting second dimension
        [(128, 128), (128, 1)],  # Broadcasting first dimension
        [(128, 128, 64), (128, 1, 64)],  # 3D tensors with broadcasting
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("test_fn", hoisted_binary_ops)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_binary_ops(
    test_fn: Callable, shapes: List[Shape], dtype: torch.dtype, request, target: str
):
    """Test binary ops that support CPU hoisting"""
    compile_ttir_to_flatbuffer(
        test_fn,
        shapes,
        [dtype] * len(shapes),
        test_base=f"{request.node.name}",
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Test hoisted permute separately because it requires unique input shapes.
@x86_only
@pytest.mark.parametrize(
    "shapes_and_perms",
    [
        # [(input_shape, output_shape), permutation]
        [[(2, 3, 4), (4, 2, 3)], [2, 0, 1]],
        [[(128, 128), (128, 128)], [0, 1]],
        [[(128, 64, 32), (32, 128, 64)], [2, 0, 1]],
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.fails_golden
def test_hoisted_permute(shapes_and_perms, request, target: str):
    shapes, permutation = shapes_and_perms

    def permute_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return permute(in0, in1, builder, permutation, unit_attrs=["ttir.should_hoist"])

    permute_wrapper.__name__ = "hoisted_permute"

    compile_ttir_to_flatbuffer(
        permute_wrapper,
        shapes,
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Test hoisted max separately because it requires more complex parameters combination.
@x86_only
@pytest.mark.parametrize("dim_arg", [None, 0, 1])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 10), (10, 1), (64, 32), (128, 64), (128, 128)]
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_max(shape, dim_arg, keep_dim, request, target: str):
    def max(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.max(
            in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=["ttir.should_hoist"]
        )

    max.__name__ = "hoisted_max"
    compile_ttir_to_flatbuffer(
        max,
        [shape],
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shape,begins,ends,step",
    [
        ((64, 64), [0, 0], [32, 32], None),
        ((128, 128), [10, 20], [50, 60], [1, 1]),
        ((32, 64, 64), [5, 10, 15], [25, 50, 55], [2, 2, 1]),
    ],
    ids=["basic_slice", "explicit_step", "3d_slice"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_slice(
    shape: Shape,
    begins: List[int],
    ends: List[int],
    step: List[int],
    target: str,
    request,
):
    def slice_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        # Now use the slice operation with the CPU hoisting attribute
        return builder.slice(in0, begins, ends, step, unit_attrs=["ttir.should_hoist"])

    compile_ttir_to_flatbuffer(
        slice_wrapper,
        [shape],
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Add test for hoisted where operation
@x86_only
@pytest.mark.parametrize("shapes", [[(64, 64), (64, 64), (64, 64)]])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_where(shapes, request, target: str):
    def where_wrapper(condition: Operand, x: Operand, y: Operand, builder: TTIRBuilder):
        return builder.where(condition, x, y, unit_attrs=["ttir.should_hoist"])

    where_wrapper.__name__ = "hoisted_where"

    compile_ttir_to_flatbuffer(
        where_wrapper,
        shapes,
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # [input_shape, output_shape]
        [(128, 128), (16384,)],  # Flatten 2D to 1D
        [(24,), (2, 3, 4)],  # Unflatten 1D to 3D
        [(2, 3, 4), (6, 4)],  # 3D to 2D reshape
        [(128, 128), (64, 256)],  # 2D to 2D different arrangement
        [(1, 1, 1), (1,)],  # Edge case: all dimensions are 1
        [(10,), (10,)],  # Identity reshape
        [(64, 512), (64, 1, 512)],  # Common ML pattern: expand dims
        [(256, 256), (512, 128)],  # Power of 2 reshape
        [(32, 3, 224, 224), (32, 150528)],  # Large ML pattern: batch flatten
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
def test_reshape(shapes, dtype: torch.dtype, request):
    input_shape, output_shape = shapes

    def reshape_wrapper(in0: Operand, builder: TTIRBuilder):
        return builder.reshape(in0, output_shape)

    compile_ttir_to_flatbuffer(
        reshape_wrapper,
        [input_shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        # [input_shape, output_shape]
        [(2, 3, 4), (24,)],
        [(128, 128), (16384,)],
        [(128, 64, 32), (128, 2048)],
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_reshape(shapes, request, target: str):
    input_shape, output_shape = shapes

    def reshape_wrapper(in0: Operand, builder: TTIRBuilder):
        return builder.reshape(in0, output_shape, unit_attrs=["ttir.should_hoist"])

    reshape_wrapper.__name__ = "hoisted_reshape"

    compile_ttir_to_flatbuffer(
        reshape_wrapper,
        [input_shape],
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes_and_dims",
    [
        # [(input_shape, output_shape), permutation]
        [[(2, 3, 4), (2, 4, 3)], [2, 1]],
        [[(128, 128), (128, 128)], [1, 0]],
        [[(128, 64, 32), (32, 64, 128)], [0, 2]],
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_transpose(shapes_and_dims, request, target: str):
    shapes, dims = shapes_and_dims

    def transpose_wrapper(in0: Operand, builder: TTIRBuilder):
        # For 2D tensors with permutation [1, 0], swap dimensions 0 and 1
        # For 3D tensors with permutation [2, 1, 0], swap dimensions 0 and 2
        dim0 = dims[0]
        dim1 = dims[1]
        return builder.transpose(
            in0, dim0=dim0, dim1=dim1, unit_attrs=["ttir.should_hoist"]
        )

    transpose_wrapper.__name__ = "hoisted_transpose"

    compile_ttir_to_flatbuffer(
        transpose_wrapper,
        [shapes[0]],
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


unary_ops = [
    exp | Marks(pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4078")),
    expm1 | Marks(pytest.mark.skip_config(["ttmetal"])),
    floor | Marks(pytest.mark.fails_golden),
    abs,
    neg,
    sign | Marks(pytest.mark.skip_config(["ttmetal"])),
    cos | Marks(pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4083")),
    sin | Marks(pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4083")),
    atan | Marks(pytest.mark.skip_config(["ttmetal"])),
    tanh | Marks(pytest.mark.skip_config(["ttmetal"])),
    relu | Marks(pytest.mark.skip_config(["ttmetal"])),
    gelu | Marks(pytest.mark.skip_config(["ttmetal"])),
    leaky_relu | Marks(pytest.mark.skip_config(["ttmetal"])),
    cbrt | Marks(pytest.mark.skip_config(["ttmetal"])),
    sigmoid | Marks(pytest.mark.fails_golden),
    reciprocal,
    is_finite | Marks(pytest.mark.skip_config(["ttmetal"])),
    ceil | Marks(pytest.mark.skip_config(["ttmetal"])),
    sum | Marks(pytest.mark.skip_config(["ttmetal"])),
    mean | Marks(pytest.mark.skip_config(["ttmetal"])),
    max | Marks(pytest.mark.fails_golden, pytest.mark.skip_config(["ttmetal"])),
    min | Marks(pytest.mark.fails_golden, pytest.mark.skip_config(["ttmetal"])),
    get_dimension_size
    | Marks(
        pytest.mark.skip_config(["ttmetal"]),
        pytest.mark.skip_config(["ttnn-standalone"]),
    ),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "ttnn-standalone"])
@pytest.mark.parametrize("test_fn", unary_ops)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
):
    pipeline_options = []
    compile_ttir_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        add,
        multiply,
        subtract,
        eq | Marks(pytest.mark.skip_config(["ttmetal"])),
        ne | Marks(pytest.mark.skip_config(["ttmetal"])),
        le | Marks(pytest.mark.skip_config(["ttmetal"])),
        lt | Marks(pytest.mark.skip_config(["ttmetal"])),
        ge | Marks(pytest.mark.skip_config(["ttmetal"])),
        gt | Marks(pytest.mark.skip_config(["ttmetal"])),
        remainder | Marks(pytest.mark.skip_config(["ttmetal"])),
        maximum
        | Marks(pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4084")),
        minimum | Marks(pytest.mark.skip_config(["ttmetal"])),
        matmul | Marks(pytest.mark.skip_config(["ttmetal"])),
        logical_and | Marks(pytest.mark.skip_config(["ttmetal"])),
        logical_or | Marks(pytest.mark.skip_config(["ttmetal"])),
        logical_xor | Marks(pytest.mark.skip_config(["ttmetal"])),
    ],
)
def test_binary_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    # NOTE: this function is _only_ for binary ops that take the same shape arguments
    pipeline_options = []
    compile_ttir_to_flatbuffer(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


@pytest.mark.run_error
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("test_fn", [bitwise_and, bitwise_or, bitwise_xor])
def test_bitwise_binary_ops(test_fn: Callable, shape: Shape, request):
    compile_ttir_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape] * 2,
        inputs_types=[torch.int8] * 2,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Subtract and remainder ops do not support broadcasting on both operands.
# This is tracked in the following Metal issue: https://github.com/tenstorrent/tt-metal/issues/24635.
# There are operations that still do not support Int32 tracked here: https://github.com/tenstorrent/tt-metal/issues/25112.
@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param([(1, 1, 1), (8, 16, 32)], id="broadcast_lhs_1"),
        pytest.param([(1, 1, 32), (8, 16, 32)], id="broadcast_lhs_2"),
        pytest.param([(1, 16, 32), (8, 16, 32)], id="broadcast_lhs_3"),
        pytest.param([(8, 16, 32), (1, 1, 1)], id="broadcast_rhs_1"),
        pytest.param([(8, 16, 32), (1, 1, 32)], id="broadcast_rhs_2"),
        pytest.param([(8, 16, 32), (1, 16, 32)], id="broadcast_rhs_3"),
        pytest.param([(8, 16, 1), (1, 1, 32)], id="broadcast_both_1"),
        pytest.param([(1, 1, 32), (8, 16, 1)], id="broadcast_both_2"),
        pytest.param([(8, 1, 32), (8, 16, 1)], id="broadcast_both_3"),
        pytest.param([(8, 16, 1), (8, 1, 32)], id="broadcast_both_4"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        add | Marks(pytest.mark.run_error),
        multiply | Marks(pytest.mark.run_error),
        subtract | Marks(pytest.mark.run_error),
        eq | Marks(pytest.mark.run_error),
        ne,
        le,
        lt,
        ge,
        gt,
        div | Marks(pytest.mark.run_error),
        remainder | Marks(pytest.mark.run_error),
        maximum,
        minimum,
        pow | Marks(pytest.mark.run_error),
        logical_and,
        logical_or,
        logical_xor,
    ],
)
def test_binary_eltwise_ops_implicit_broadcast(
    test_fn: Callable,
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
):
    compile_ttir_to_flatbuffer(
        test_fn,
        shapes,
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 16, 32), (8, 16, 32), (8, 16, 32)],
        [(8, 16, 32), (1, 16, 32), (8, 16, 32)],
        [(8, 16, 32), (8, 16, 32), (1, 16, 32)],
        [(8, 16, 32), (1, 1, 32), (1, 1, 32)],
        [(1, 1, 32), (8, 16, 32), (1, 1, 32)],
        [(1, 1, 32), (1, 1, 32), (8, 16, 32)],
        [(1, 16, 32), (8, 1, 32), (8, 16, 1)],
        [(1, 4, 1), (1, 4, 768), (1, 1, 1)],
        [(1, 1, 1, 4), (1, 1, 1, 1), (1, 1, 1, 1)],
    ],
)
@pytest.mark.parametrize(
    "input_dtypes",
    [
        pytest.param((torch.float32, torch.float32, torch.float32), id="f32-f32-f32"),
        pytest.param((torch.float32, torch.int32, torch.int32), id="f32-i32-i32"),
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", [where])
def test_ternary_eltwise_ops_implicit_broadcast(
    test_fn: Callable,
    shapes: List[Shape],
    input_dtypes: Tuple[torch.dtype, torch.dtype, torch.dtype],
    target: str,
    request,
):
    dtype1, dtype2, dtype3 = input_dtypes

    compile_ttir_to_flatbuffer(
        test_fn,
        shapes,
        [dtype1, dtype2, dtype3],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@pytest.mark.parametrize(
    "test_fn,inputs_shapes,inputs_dtypes",
    [
        (transpose, [(64, 32)], None),
        pytest.param(
            embedding,
            [(33, 32), (512, 128)],
            [torch.float32] * 2,
        ),
        pytest.param(
            where,
            [(64, 64)] * 3,
            [torch.float32, torch.float32, torch.float32],
            marks=pytest.mark.fails_golden,
        ),
    ],
)
def test_unique_ops(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_dtypes: List[torch.dtype],
    request,
):
    compile_ttir_to_flatbuffer(
        test_fn,
        inputs_shapes=inputs_shapes,
        inputs_types=inputs_dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def slice(
    in0: Operand,
    begins: List[int],
    ends: List[int],
    step: Optional[List[int]] = None,
    builder: TTIRBuilder = None,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.slice(in0, begins, ends, step, unit_attrs=unit_attrs)


@pytest.mark.parametrize(
    "shape,begins,ends,step",
    [
        ((64, 64), [0, 0], [32, 32], None),
        ((64, 64), [10, 20], [50, 60], [1, 1]),
        ((64, 64, 64), [10, 20, 30], [50, 60, 64], [2, 2, 1]),
    ],
    ids=["basic_slice", "explicit_step", "3d_slice"],
)
def test_slice(
    shape: Shape, begins: List[int], ends: List[int], step: List[int], request
):
    def slice_op(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return slice(in0, begins, ends, step, builder, unit_attrs)

    compile_ttir_to_flatbuffer(
        slice_op,
        [shape],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0]])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.skip("Runtime error: https://github.com/tenstorrent/tt-mlir/issues/3883")
def test_hoisted_reduce_or(shape: Shape, dim_args: List[int], target: str, request):
    """Test the hoisted reduce_or operation with proper dimensions and keep_dim parameter"""

    def hoisted_reduce_or_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return reduce_or(
            in0, builder, dim_args, keep_dim=True, unit_attrs=["ttir.should_hoist"]
        )

    compile_ttir_to_flatbuffer(
        hoisted_reduce_or_wrapper,
        inputs_shapes=[shape],
        inputs_types=[torch.float32],
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes_and_broadcast_dims",
    [
        # [(input_shape, output_shape), broadcast_dimensions]
        [[(1, 1, 32), (1, 16, 32)], [1, 16, 1]],
        [[(128, 1), (128, 64)], [1, 64]],
        [[(1, 128), (64, 128)], [64, 1]],
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_broadcast(shapes_and_broadcast_dims, request, target: str):
    """Test broadcast operation with CPU hoisting enabled using the 'hoisted_' naming convention"""
    shapes, broadcast_dimensions = shapes_and_broadcast_dims

    def broadcast_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return broadcast(
            in0, in1, builder, broadcast_dimensions, unit_attrs=["ttir.should_hoist"]
        )

    broadcast_wrapper.__name__ = "hoisted_broadcast"

    compile_ttir_to_flatbuffer(
        broadcast_wrapper,
        inputs_shapes=shapes,
        test_base=f"{request.node.name}",
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def gather(
    in0: Operand,
    builder: TTIRBuilder,
    indices_shape: Shape,
    start_index_map: List[int],
    offset_dims: List[int],
    slice_sizes: List[int],
    indices_dtype: torch.dtype,
    unit_attrs: Optional[List[str]] = None,
):
    # For now, just create zero indices - this tests the basic gather functionality.
    # In a real test, you'd want to create varied indices to test different gather patterns.
    indices = builder.zeros(indices_shape, indices_dtype)

    # Set collapsed_slice_dims to be the same as start_index_map
    # This is what the GatherToEmbeddingConversionPattern expects.
    collapsed_slice_dims = start_index_map

    # Set remaining parameters to empty lists for simplicity.
    operand_batching_dims = []
    start_indices_batching_dims = []

    # Set index_vector_dim correctly based on the use case.
    if len(indices_shape) == 1 and len(start_index_map) == 1:
        # Single indices case - index vector dim is implicit.
        index_vector_dim = len(indices_shape)  # = 1
    else:
        # Multi-dimensional indices - last dimension contains index vectors.
        index_vector_dim = len(indices_shape) - 1

    return builder.gather(
        in0,
        indices,
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        operand_batching_dims=operand_batching_dims,
        start_indices_batching_dims=start_indices_batching_dims,
        start_index_map=start_index_map,
        index_vector_dim=index_vector_dim,
        slice_sizes=slice_sizes,
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize(
    "input_shape,input_dtype,indices_shape,start_index_map,offset_dims,slice_sizes",
    [
        # Simple 1D indices - f32.
        ((100, 50), torch.float32, (10,), [0], [1], [1, 50]),
        pytest.param(
            (8, 16, 32),
            torch.float32,
            (4, 2, 2),
            [0, 2],
            [1],
            # Complex indices - f32.
            [1, 16, 1],
            marks=pytest.mark.skip(
                reason="Multi-dimensional gather has known issues, but the builder golden may also be incorrect: https://github.com/tenstorrent/tt-mlir/issues/3884"
            ),
        ),
        pytest.param(
            (8, 16, 32),
            torch.bfloat16,
            (4, 2, 2),
            [0, 2],
            [1],
            # Complex indices - bf16.
            [1, 16, 1],
            marks=pytest.mark.skip(
                reason="Multi-dimensional gather has known issues, but the builder golden may also be incorrect: https://github.com/tenstorrent/tt-mlir/issues/3884"
            ),
        ),
    ],
    ids=[
        "simple_1d-f32",
        "complex_indices-f32",
        "complex_indices-bf16",
    ],
)
def test_gather(
    input_shape: Shape,
    input_dtype: torch.dtype,
    indices_shape: Shape,
    start_index_map: List[int],
    offset_dims: List[int],
    slice_sizes: List[int],
    request,
):
    def gather_wrapper(in0: Operand, builder: TTIRBuilder):
        return gather(
            in0,
            builder,
            indices_shape,
            start_index_map,
            offset_dims,
            slice_sizes,
            input_dtype,
        )

    compile_ttir_to_flatbuffer(
        gather_wrapper,
        [input_shape],
        [input_dtype],
        test_base=request.node.name,
        target="ttnn",
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "input_shape,indices_shape,start_index_map,offset_dims,slice_sizes",
    [
        ((100, 50), (10,), [0], [1], [1, 50]),  # Simple 1D indices
        (
            (8, 16, 32),
            (4, 2, 2),
            [0, 2],
            [1],
            [1, 16, 1],
        ),  # Complex indices)
    ],
    ids=["simple_1d", "complex_indices"],
)
# Note: Doesn't work on ttmetal because test generated (nonhoisted) ttir.zeros, which we need to support on device.
@pytest.mark.skip(
    "Fails at runtime on simple_1d case, ticket: https://github.com/tenstorrent/tt-mlir/issues/3849"
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_gather(
    input_shape: Shape,
    indices_shape: Shape,
    start_index_map: List[int],
    offset_dims: List[int],
    slice_sizes: List[int],
    target: str,
    request,
):
    def gather_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return gather(
            in0,
            builder,
            indices_shape,
            start_index_map,
            offset_dims,
            slice_sizes,
            unit_attrs=["ttir.should_hoist"],
        )

    compile_ttir_to_flatbuffer(
        gather_wrapper,
        [input_shape],
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
    [
        # Standard matrix multiplication: [M, K] x [K, N] -> [M, N]
        ([(10, 20), (20, 30), (10, 30)], [], [1], [], [0]),
        # Batched matrix multiplication: [B, M, K] x [B, K, N] -> [B, M, N]
        ([(5, 10, 20), (5, 20, 30), (5, 10, 30)], [0], [2], [0], [1]),
        # 3D tensor @ 2D tensor: [B, M, K] x [K, N] -> [B, M, N]
        ([(5, 10, 20), (20, 30), (5, 10, 30)], [], [2], [], [0]),
    ],
    ids=["standard_matmul", "batched_matmul", "3d_tensor_2d_tensor"],
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.skip(
    "Need to rework this, https://github.com/tenstorrent/tt-mlir/issues/3851"
)
def test_hoisted_dot_general(
    shapes: List[Shape],
    batch_dims_lhs: List[int],
    contract_dims_lhs: List[int],
    batch_dims_rhs: List[int],
    contract_dims_rhs: List[int],
    target: str,
    request,
):
    def dot_general_wrapper(
        in0: Operand,
        in1: Operand,
        out0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.dot_general(
            in0,
            in1,
            out0,
            batch_dims_lhs,
            contract_dims_lhs,
            batch_dims_rhs,
            contract_dims_rhs,
            unit_attrs=["ttir.should_hoist"],
        )

    compile_ttir_to_flatbuffer(
        dot_general_wrapper,
        shapes,
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
