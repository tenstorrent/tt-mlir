# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional
from conftest import x86_only
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import (
    compile_and_execute_ttir,
)
from test_utils import (
    Marks,
    shape_str,
)

pytestmark = pytest.mark.frontend("ttir")

# Unary ops
def abs(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.abs(in0, unit_attrs=unit_attrs)


def atan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.atan(in0, unit_attrs=unit_attrs)


def cbrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cbrt(in0, unit_attrs=unit_attrs)


def ceil(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.ceil(in0, unit_attrs=unit_attrs)


def erf(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.erf(in0, unit_attrs=unit_attrs)


def erfc(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.erfc(in0, unit_attrs=unit_attrs)


def exp(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.exp(in0, unit_attrs=unit_attrs)


def expm1(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.expm1(in0, unit_attrs=unit_attrs)


def floor(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.floor(in0, unit_attrs=unit_attrs)


def cos(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cos(in0, unit_attrs=unit_attrs)


def gelu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.gelu(in0, unit_attrs=unit_attrs)


def is_finite(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.is_finite(in0, unit_attrs=unit_attrs)


# Special handling for log PCC checks. Due to the vertical asymptote on the log graph,
# small changes in input values result in large changes in output values at negative values,
# so both graph and golden tensors must be constrained accordingly.
def log(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    log_0 = builder.log(in0, unit_attrs=unit_attrs)

    # Constrain values for log
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("log op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    abs_tensor = torch.abs(randn_tensor)
    error_margin = torch.full(randn_tensor.shape, 0.01)
    input_golden = torch.add(abs_tensor, error_margin)
    output_golden = torch.log(input_golden)
    builder.set_goldens({in0: input_golden}, {log_0: output_golden})
    return log_0


def log1p(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    log1p_0 = builder.log1p(in0, unit_attrs=unit_attrs)

    # Constrain values for log1p
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("log1p op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    abs_tensor = torch.abs(randn_tensor)
    error_margin = torch.full(randn_tensor.shape, -0.99)
    input_golden = torch.add(abs_tensor, error_margin)
    output_golden = torch.log1p(input_golden)

    builder.set_goldens({in0: input_golden}, {log1p_0: output_golden})
    return log1p_0


def logical_not(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.logical_not(in0, unit_attrs=unit_attrs)


def neg(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.neg(in0, unit_attrs=unit_attrs)


def relu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.relu(in0, unit_attrs=unit_attrs)


def reciprocal(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    reciprocal_0 = builder.reciprocal(in0, unit_attrs=unit_attrs)

    # Constrain values for reciprocal
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("reciprocal op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    input = torch.abs(torch.randn(in0.type.shape, dtype=dtype))
    input_safe = torch.clamp(input, min=-1e-6, max=None)
    input_safe = torch.where(input_safe == 0, torch.tensor(1e-6), input_safe)
    golden_output = torch.reciprocal(input_safe)
    builder.set_goldens({in0: input_safe}, {reciprocal_0: golden_output})
    return reciprocal_0


def relu6(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.relu6(in0, unit_attrs=unit_attrs)


def rsqrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    rsqrt_0 = builder.rsqrt(in0, unit_attrs=unit_attrs)
    # Constrain values for rsqrt
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("rsqrt op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    input_tensor = torch.abs(torch.randn(in0.type.shape, dtype=dtype))
    golden_output_tensor = torch.rsqrt(input_tensor)
    builder.set_goldens({in0: input_tensor}, {rsqrt_0: golden_output_tensor})
    return rsqrt_0


def sigmoid(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sigmoid(in0, unit_attrs=unit_attrs)


def sign(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sign(in0, unit_attrs=unit_attrs)


def silu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.silu(in0, unit_attrs=unit_attrs)


def sqrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    sqrt_0 = builder.sqrt(in0, unit_attrs=unit_attrs)

    # Constrain values for sqrt
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("rsqrt op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    input_tensor = torch.abs(torch.randn(in0.type.shape, dtype=dtype))
    golden_output_tensor = torch.sqrt(input_tensor)
    builder.set_goldens({in0: input_tensor}, {sqrt_0: golden_output_tensor})
    return sqrt_0


def sin(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sin(in0, unit_attrs=unit_attrs)


# Special handling for log PCC checks. Due to the vertical asymptote on the log graph,
# small changes in input values result in large changes in output values at negative values,
# so both graph and golden tensors must be constrained accordingly.
def tan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    tan_0 = builder.tan(in0, unit_attrs=unit_attrs)

    # Constrain values for tan
    import math

    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("tan op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    input_golden = randn_tensor.uniform_((-math.pi / 2 + 0.05), (math.pi / 2 - 0.05))
    output_golden = torch.tan(input_golden)
    builder.set_goldens({in0: input_golden}, {tan_0: output_golden})
    return tan_0


def tanh(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.tanh(in0, unit_attrs=unit_attrs)


unary_ops = [
    abs,
    atan | Marks(pytest.mark.skip_config(["ttmetal"])),
    cbrt | Marks(pytest.mark.skip_config(["ttmetal"])),
    ceil | Marks(pytest.mark.skip_config(["ttmetal"])),
    cos,
    erf | Marks(pytest.mark.skip_config(["ttmetal"])),
    erfc | Marks(pytest.mark.skip_config(["ttmetal"])),
    exp,
    expm1 | Marks(pytest.mark.skip_config(["ttmetal"])),
    floor,
    gelu,
    is_finite | Marks(pytest.mark.skip_config(["ttmetal"])),
    log,
    log1p | Marks(pytest.mark.skip_config(["ttmetal"])),
    logical_not,  # TODO (wenbinlyuTT): test int32 once untilize issue is fixed
    neg,
    reciprocal,
    relu,
    relu6 | Marks(pytest.mark.skip_config(["ttmetal"])),
    rsqrt,
    sigmoid,
    sign | Marks(pytest.mark.skip_config(["ttmetal"])),
    silu,
    sin,
    sqrt,
    tan,
    tanh | Marks(pytest.mark.skip_config(["ttmetal"])),
]


unary_ops_dtypes = [
    torch.float32,
    torch.int32 | Marks(pytest.mark.skip_config(["ttmetal"])),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", unary_ops_dtypes, ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitc", "emitpy"])
@pytest.mark.parametrize("test_fn", unary_ops)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if dtype == torch.int32 and test_fn not in [
        abs,
        neg,
        relu,
    ]:
        pytest.skip("int32 unary op is not supported yet for this operation")

    pipeline_options = []
    compile_and_execute_ttir(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


# Bitwise unary ops (int only)
def bitwise_not(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.bitwise_not(in0, unit_attrs=unit_attrs)


bitwise_unary_ops = [bitwise_not]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
@pytest.mark.parametrize("test_fn", bitwise_unary_ops)
def test_bitwise_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if target == "ttmetal":
        pytest.xfail(reason="i32 unary ops not supported on ttmetal yet")
    compile_and_execute_ttir(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# Unary ops with float parameter
def leaky_relu(
    in0: Operand,
    parameter: float,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.leaky_relu(in0, parameter, unit_attrs=unit_attrs)


unary_ops_with_float_param = [leaky_relu | Marks(pytest.mark.skip_config(["ttmetal"]))]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitc", "emitpy"])
@pytest.mark.parametrize("test_fn", unary_ops_with_float_param)
@pytest.mark.parametrize("parameter", [0.01, 0.1, 0.2])
def test_unary_ops_with_float_param(
    test_fn: Callable,
    parameter: float,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def wrapper_func(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return test_fn(in0, parameter, builder, unit_attrs=unit_attrs)

    pipeline_options = []
    compile_and_execute_ttir(
        wrapper_func,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


# Unary ops with int parameter
def get_dimension_size(
    in0: Operand,
    parameter: int,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.get_dimension_size(in0, parameter, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize("dimension", [0, 1])
def test_get_dimension_size(
    dimension: int,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def wrapper_func(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return get_dimension_size(in0, dimension, builder, unit_attrs=unit_attrs)

    pipeline_options = []
    compile_and_execute_ttir(
        wrapper_func,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


# Unaligned shapes tests for neg op
unaligned_shapes = [
    (5, 3),
    (32, 1),
    (31, 7),
    (1, 32),
    (13, 29),
    (64, 1),
    (61, 3),
    (61, 37),
    (1, 64),
    (5, 67),
    (43, 67),
    (2, 3, 5),
    (3, 17, 37),
    (9, 43, 7),
    (5, 61, 49),
    (51, 19, 23) | Marks(pytest.mark.xfail(reason="Golden failure")),
    (677, 1, 1) | Marks(pytest.mark.xfail(reason="Golden failure")),
    (2, 3, 5, 7),
    (3, 37, 5, 53) | Marks(pytest.mark.xfail(reason="Golden failure")),
    (37, 3, 5, 53) | Marks(pytest.mark.xfail(reason="Golden failure")),
    (41, 7, 43, 11),
    (7, 41, 43, 11),
    (1, 23, 1, 1),
    (23, 1, 1, 1),
    (3, 5, 7, 11, 13),
]


@pytest.mark.parametrize("shape", unaligned_shapes, ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unaligned_shapes_neg(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_ttir(
        neg,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# Hoisted unary ops

# Create hoisted versions of operations by currying the unit_attrs parameter
def create_hoisted_unary_op(op_func, name):
    """Create a hoisted version of a unary operation by adding the should_hoist unit attribute"""

    def hoisted_op(in0, builder, **kwargs):
        # For unary ops
        return op_func(in0, builder, unit_attrs=["ttir.should_hoist"], **kwargs)

    # Set the name for better test identification
    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


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
    create_hoisted_unary_op(relu, "relu"),
]


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("test_fn", hoisted_unary_ops)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_unary_ops(
    test_fn: Callable,
    shape: Shape,
    request,
    target: str,
    device,
    dtype: torch.dtype = torch.float32,
):
    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttir(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=f"{request.node.name}",
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
