# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# ttmetal-only mirrors of multi-backend tests in
# `ttir_ops/eltwise/test_ttir_unary.py`. Helper functions are inlined here
# so the d2m mirrors are self-contained.

import math

import pytest
import torch
from typing import Callable, List, Optional

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from d2m.shape_cases import ELEMENTWISE_SHAPES, UNALIGNED_SHAPES, rotated_params
from test_utils import Marks, SkipIf, shape_str

pytestmark = pytest.mark.frontend("ttir")


# ---------------------------------------------------------------------------
# Unary op helpers (mirrored from ttir_ops/eltwise/test_ttir_unary.py).
# ---------------------------------------------------------------------------
def abs(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.abs(in0, unit_attrs=unit_attrs)


def acos(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    acos_0 = builder.acos(in0, unit_attrs=unit_attrs)
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("acos op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if str(in0.type.element_type) == "bf16" else torch.float32
    rand = torch.rand(in0.type.shape, dtype=dtype) * 2 - 1
    input_golden = rand * 0.999
    output_golden = torch.acos(input_golden)
    builder.set_goldens({in0: input_golden}, {acos_0: output_golden})
    return acos_0


def asin(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    asin_0 = builder.asin(in0, unit_attrs=unit_attrs)
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("asin op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if str(in0.type.element_type) == "bf16" else torch.float32
    rand = torch.rand(in0.type.shape, dtype=dtype) * 2 - 1
    input_golden = rand * 0.999
    output_golden = torch.asin(input_golden)
    builder.set_goldens({in0: input_golden}, {asin_0: output_golden})
    return asin_0


def asinh(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.asinh(in0, unit_attrs=unit_attrs)


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


def log(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    log_0 = builder.log(in0, unit_attrs=unit_attrs)
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("log op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if str(in0.type.element_type) == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    abs_tensor = torch.abs(randn_tensor)
    error_margin = torch.full(randn_tensor.shape, 0.01, dtype=dtype)
    input_golden = torch.add(abs_tensor, error_margin)
    output_golden = torch.log(input_golden)
    builder.set_goldens({in0: input_golden}, {log_0: output_golden})
    return log_0


def log1p(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    log1p_0 = builder.log1p(in0, unit_attrs=unit_attrs)
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("log1p op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if str(in0.type.element_type) == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    abs_tensor = torch.abs(randn_tensor)
    error_margin = torch.full(randn_tensor.shape, -0.99, dtype=dtype)
    input_golden = torch.add(abs_tensor, error_margin)
    output_golden = torch.log1p(input_golden)
    builder.set_goldens({in0: input_golden}, {log1p_0: output_golden})
    return log1p_0


def logical_not(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.logical_not(in0, unit_attrs=unit_attrs)


def mish(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.mish(in0, unit_attrs=unit_attrs)


def neg(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.neg(in0, unit_attrs=unit_attrs)


def relu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.relu(in0, unit_attrs=unit_attrs)


def reciprocal(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    reciprocal_0 = builder.reciprocal(in0, unit_attrs=unit_attrs)
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("reciprocal op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if str(in0.type.element_type) == "bf16" else torch.float32
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
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("rsqrt op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if str(in0.type.element_type) == "bf16" else torch.float32
    input_tensor = torch.abs(torch.randn(in0.type.shape, dtype=dtype))
    golden_output_tensor = torch.rsqrt(input_tensor)
    builder.set_goldens({in0: input_tensor}, {rsqrt_0: golden_output_tensor})
    return rsqrt_0


def sigmoid(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sigmoid(in0, unit_attrs=unit_attrs)


def hardsigmoid(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.hardsigmoid(in0, unit_attrs=unit_attrs)


def sign(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sign(in0, unit_attrs=unit_attrs)


def silu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.silu(in0, unit_attrs=unit_attrs)


def sqrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    sqrt_0 = builder.sqrt(in0, unit_attrs=unit_attrs)
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("rsqrt op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if str(in0.type.element_type) == "bf16" else torch.float32
    input_tensor = torch.abs(torch.randn(in0.type.shape, dtype=dtype))
    golden_output_tensor = torch.sqrt(input_tensor)
    builder.set_goldens({in0: input_tensor}, {sqrt_0: golden_output_tensor})
    return sqrt_0


def square(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.square(in0, unit_attrs=unit_attrs)


def exp2(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.exp2(in0, unit_attrs=unit_attrs)


def softsign(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.softsign(in0, unit_attrs=unit_attrs)


def signbit(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.signbit(in0, unit_attrs=unit_attrs)


def selu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.selu(in0, unit_attrs=unit_attrs)


def frac(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.frac(in0, unit_attrs=unit_attrs)


def trunc(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.trunc(in0, unit_attrs=unit_attrs)


def sin(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sin(in0, unit_attrs=unit_attrs)


def tan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    tan_0 = builder.tan(in0, unit_attrs=unit_attrs)
    if str(in0.type.element_type) not in ["bf16", "f32"]:
        raise ValueError("tan op only supports bf16 and f32 data types")
    dtype = torch.bfloat16 if str(in0.type.element_type) == "bf16" else torch.float32
    randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
    input_golden = randn_tensor.uniform_((-math.pi / 2 + 0.05), (math.pi / 2 - 0.05))
    output_golden = torch.tan(input_golden)
    builder.set_goldens({in0: input_golden}, {tan_0: output_golden})
    return tan_0


def tanh(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.tanh(in0, unit_attrs=unit_attrs)


unary_ops = [
    abs,
    acos | Marks(pytest.mark.skip_config(["ttmetal"])),
    asin | Marks(pytest.mark.skip_config(["ttmetal"])),
    asinh | Marks(pytest.mark.skip_config(["ttmetal"])),
    atan | Marks(pytest.mark.skip_config(["ttmetal"])),
    cbrt | Marks(pytest.mark.skip_config(["ttmetal"])),
    ceil,
    cos,
    erf,
    erfc,
    exp,
    expm1,
    exp2 | SkipIf("ttnn", "emitc", "emitpy", "sim"),
    floor,
    gelu,
    is_finite | Marks(pytest.mark.skip_config(["ttmetal"])),
    log,
    log1p,
    logical_not,
    mish | Marks(pytest.mark.skip_config(["ttmetal"])),
    neg,
    reciprocal,
    relu,
    relu6 | Marks(pytest.mark.skip_config(["ttmetal"])),
    rsqrt,
    sigmoid,
    sign,
    hardsigmoid,
    silu,
    sin,
    sqrt,
    square | SkipIf("ttnn", "emitc", "emitpy", "sim"),
    softsign | SkipIf("ttnn", "emitc", "emitpy", "sim"),
    signbit | SkipIf("ttnn", "emitc", "emitpy", "sim"),
    selu | SkipIf("ttnn", "emitc", "emitpy", "sim"),
    frac | SkipIf("ttnn", "emitc", "emitpy", "sim"),
    trunc | SkipIf("ttnn", "emitc", "emitpy", "sim"),
    tan,
    tanh,
]

unary_ops_float_dtypes = [
    torch.float32,
    torch.bfloat16,
]

unary_ops_i32 = [
    abs,
    erf,
    logical_not,
    neg,
    relu,
    sign,
]


def _compile_unary_op(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if dtype == torch.int32 and getattr(test_fn, "__name__", None) not in {
        "abs",
        "erf",
        "is_finite",
        "logical_not",
        "neg",
        "relu",
        "sign",
    }:
        pytest.skip("int32 unary op is not in the allowlist for this test")

    # tt-metal #41850 replaced the SFPU erf kernel with a LUT-based rational
    # approximation. The new kernel reads the input as a float, so int32 bit
    # patterns become NaN/Inf and the output diverges from torch.erf. The
    # TTNN pipeline inserts a bf16 typecast workaround around ttnn.erf for
    # integer inputs (see TTNNWorkaroundsPass), but the TTMetal pipeline
    # lowers ttir.erf directly into a D2M tile op without that workaround.
    # Skip until a TTIR-level decomposition is added for ttmetal.
    # Tracking issue: https://github.com/tenstorrent/tt-mlir/issues/8105
    if dtype == torch.int32 and test_fn is erf and target == "ttmetal":
        pytest.skip(
            "erf with int32 input is not supported on ttmetal yet; "
            "TTNN typecast workaround doesn't apply to the ttmetal pipeline. "
            "See https://github.com/tenstorrent/tt-mlir/issues/8105"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def unary_op(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return test_fn(in0, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[],
    )


_UNARY_OP_PARAMS = (
    rotated_params(
        unary_ops, ELEMENTWISE_SHAPES, unary_ops_float_dtypes, value_order=[1, 2, 0]
    )
    + rotated_params(
        unary_ops_i32,
        ELEMENTWISE_SHAPES,
        [torch.int32 | SkipIf("sim")],
        value_order=[1, 2, 0],
    )
    + [
        pytest.param((128,), torch.float32, neg, id="128-f32-neg_1d"),
    ]
)


@pytest.mark.parametrize("shape,dtype,test_fn", _UNARY_OP_PARAMS)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unary_ops(
    shape: Shape, dtype: torch.dtype, test_fn: Callable, target: str, request, device
):
    _compile_unary_op(test_fn, shape, dtype, target, request, device)


def _compile_collapse_tensors_case(
    test_func: Callable,
    test_name: str,
    collapse_tensors: bool,
    target: str,
    request,
    device,
):
    pipeline_options = f"{{collapse-tensors-2d={str(collapse_tensors).lower()}}}"
    pipeline = f"ttir-to-ttmetal-pipeline{pipeline_options}"

    compile_and_execute_ttir(
        test_func,
        target=target,
        custom_pipeline=pipeline,
        test_base=f"{request.node.name}_{test_name}_{'collapsed' if collapse_tensors else 'non_collapsed'}",
        device=device,
    )


def module_unary_exp_2d_exp(builder: TTIRBuilder):
    @builder.func([(3, 32, 64)], [torch.float32])
    def unary_exp(in0: Operand, builder: TTIRBuilder):
        return builder.exp(in0)


def module_unary_exp_4d_exp(builder: TTIRBuilder):
    @builder.func([(1, 2, 32, 32)], [torch.float32])
    def unary_exp(in0: Operand, builder: TTIRBuilder):
        return builder.exp(in0)


@pytest.mark.parametrize(
    "test_func,test_name",
    [
        pytest.param(module_unary_exp_2d_exp, "3d_exp", id="3d_exp"),
        pytest.param(module_unary_exp_4d_exp, "4d_exp", id="4d_exp"),
    ],
)
@pytest.mark.parametrize(
    "collapse_tensors", [True, False], ids=["collapsed", "non_collapsed"]
)
@pytest.mark.parametrize("target", ["ttmetal"], ids=["ttmetal"])
def test_unary_ops_collapse_tensors(
    test_func: Callable,
    test_name: str,
    collapse_tensors: bool,
    target: str,
    request,
    device,
):
    _compile_collapse_tensors_case(
        test_func, test_name, collapse_tensors, target, request, device
    )


# Bitwise unary ops (int only)
def bitwise_not(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.bitwise_not(in0, unit_attrs=unit_attrs)


bitwise_unary_ops = [bitwise_not]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32 | SkipIf("sim")], ids=["i32"])
@pytest.mark.parametrize("test_fn", bitwise_unary_ops)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_bitwise_unary_ops(
    shape: Shape, dtype: torch.dtype, test_fn: Callable, target: str, request, device
):
    pytest.xfail(reason="i32 unary ops not supported on ttmetal yet")

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def bitwise_unary_ops(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return test_fn(in0, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
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


unary_ops_with_float_param = [leaky_relu | SkipIf("ttmetal")]


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("test_fn", unary_ops_with_float_param)
@pytest.mark.parametrize("parameter", [0.01, 0.1, 0.2])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unary_ops_with_float_param(
    shape: Shape,
    dtype: torch.dtype,
    test_fn: Callable,
    parameter: float,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def unary_ops_with_float_param(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return test_fn(in0, parameter, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[],
    )


@pytest.mark.parametrize("shape", UNALIGNED_SHAPES + [(677, 1, 1)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unaligned_shapes_neg(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.neg(in0, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        print_ir=False,
    )


@pytest.mark.parametrize("shape", [(512, 512)])
@pytest.mark.parametrize("target", ["ttmetal" | SkipIf("sim")])
def test_bfp8_triple_exp_f32(shape: Shape, target: str, request, device):
    pipeline_options = ["global-data-format-target=bfp_bf8"]

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def triple_exp_f32(
            in0: Operand,
            builder: TTIRBuilder,
        ):
            input_0 = torch.rand(shape, dtype=torch.float32)
            exp0 = builder.exp(in0)
            tcast0 = builder.typecast(
                exp0, torch.bfloat16, unit_attrs=["preserveDataFormat"]
            )
            exp1 = builder.exp(tcast0)
            tcast1 = builder.typecast(
                exp1, torch.float32, unit_attrs=["preserveDataFormat"]
            )
            exp2 = builder.exp(tcast1)
            output_0 = torch.exp(torch.exp(torch.exp(input_0)))
            builder.set_goldens({in0: input_0}, {exp2: output_0})
            return exp2

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
        save_artifacts=True,
        pcc=0.988,
    )


@pytest.mark.parametrize("shape", [(512, 512)])
@pytest.mark.parametrize("target", ["ttmetal" | SkipIf("sim")])
def test_bfp8_exp_f32(shape: Shape, target: str, request, device):
    pipeline_options = ["global-data-format-target=bfp_bf8"]

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def exp_f32(
            in0: Operand,
            builder: TTIRBuilder,
        ):
            input_0 = torch.rand(shape, dtype=torch.float32)
            result = builder.exp(in0)
            output_0 = torch.exp(input_0)
            builder.set_goldens({in0: input_0}, {result: output_0})
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
        save_artifacts=True,
    )


@pytest.mark.parametrize("shape", [(512, 512)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_bfp8_cos_bf16(shape: Shape, target: str, request, device):
    pipeline_options = ["global-data-format-target=bfp_bf8"]

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.bfloat16])
        def cos_bf16(
            in0: Operand,
            builder: TTIRBuilder,
        ):
            input_0 = torch.rand(shape, dtype=torch.bfloat16)
            result = builder.cos(in0)
            output_0 = torch.cos(input_0).to(torch.bfloat16)
            builder.set_goldens({in0: input_0}, {result: output_0})
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
        save_artifacts=True,
    )
