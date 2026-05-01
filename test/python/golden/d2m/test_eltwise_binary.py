# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# ttmetal-only mirrors of multi-backend tests in
# `ttir_ops/eltwise/test_ttir_binary.py`. Helper functions are inlined here
# so the d2m mirrors are self-contained.

import pytest
import torch
from typing import Callable, List, Optional, Tuple

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from d2m.shape_cases import (
    ELEMENTWISE_2D_SHAPES,
    ELEMENTWISE_SHAPES,
    UNALIGNED_SHAPES,
    rotated_params,
)
from test_utils import Marks, shape_str, SkipIf

pytestmark = pytest.mark.frontend("ttir")


# ---------------------------------------------------------------------------
# Binary op helpers (mirrored from ttir_ops/eltwise/test_ttir_binary.py).
# ---------------------------------------------------------------------------
def add(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.add(in0, in1, unit_attrs=unit_attrs)


def atan2(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.atan2(in0, in1, unit_attrs=unit_attrs)


def div(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.div(in0, in1, unit_attrs=unit_attrs)


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


def multiply(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


def pow(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    randn_base_tensor = builder._get_golden_tensor(in0)
    randn_exponent_tensor = builder._get_golden_tensor(in1)

    randn_base_tensor = randn_base_tensor.apply_shardwise(
        lambda shard: (
            shard.abs() if torch.is_floating_point(randn_exponent_tensor) else shard
        )
    )

    if torch.is_floating_point(randn_exponent_tensor):
        randn_base_tensor = torch.abs(randn_base_tensor)
    output_golden = torch.pow(randn_base_tensor, randn_exponent_tensor)
    pow0 = builder.pow(in0, in1, unit_attrs=unit_attrs)
    builder.set_goldens_from_builder_tensor(
        {in0: randn_base_tensor, in1: randn_exponent_tensor}, {pow0: output_golden}
    )
    return pow0


def remainder(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.remainder(in0, in1, unit_attrs=unit_attrs)


def subtract(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.subtract(in0, in1, unit_attrs=unit_attrs)


def gelu_backward(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.gelu_backward(in0, in1, approximate="none", unit_attrs=unit_attrs)


def gelu_backward_tanh(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.gelu_backward(in0, in1, approximate="tanh", unit_attrs=unit_attrs)


binary_ops = [
    add,
    atan2 | Marks(pytest.mark.skip_config(["ttmetal"])),
    div,
    gelu_backward | Marks(pytest.mark.skip_config(["ttmetal"])),
    gelu_backward_tanh | Marks(pytest.mark.skip_config(["ttmetal"])),
    maximum | SkipIf("sim"),
    minimum | SkipIf("sim"),
    multiply,
    pow,
    remainder | Marks(pytest.mark.skip_config(["ttmetal"])),
    subtract,
]

binary_ops_dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.int32 | SkipIf("sim"),
    torch.int64 | SkipIf("sim"),
]


_BINARY_OP_PARAMS = rotated_params(
    binary_ops, ELEMENTWISE_SHAPES, binary_ops_dtypes, value_order=[1, 2, 0]
) + [
    pytest.param((128,), torch.float32, add, id="128-f32-add_1d"),
]


@pytest.mark.parametrize(
    "shape,dtype,test_fn",
    _BINARY_OP_PARAMS,
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_binary_ops(
    shape: Shape, dtype: torch.dtype, test_fn: Callable, target: str, request, device
):
    if test_fn.__name__ == "pow" and (dtype == torch.int32 or dtype == torch.int64):
        pytest.xfail("TODO(dloke): int32 pow is not supported on ttmetal yet")
    if test_fn.__name__ == "div" and (dtype == torch.int32 or dtype == torch.int64):
        pytest.xfail(
            "TODO(dloke): int32 div is not supported on ttmetal yet, need to support floor or truncate division"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            # int64 on ttmetal is normalized to int32; use int32-range values so
            # truncation is a no-op and golden matches device output.
            if dtype == torch.int64:
                in0_golden = torch.randint(
                    -(2**31), 2**31, shape, dtype=torch.int64
                )
                in1_golden = torch.randint(
                    -(2**31), 2**31, shape, dtype=torch.int64
                )
                builder.set_goldens({in0: in0_golden, in1: in1_golden})
            return test_fn(in0, in1, builder)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        print_ir=False,
        pipeline_options=[],
    )


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


def module_elementwise_add_3d_add(builder: TTIRBuilder):
    @builder.func([(3, 32, 64), (3, 32, 64)], [torch.float32, torch.float32])
    def elementwise_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.add(in0, in1)


def module_elementwise_add_4d_add(builder: TTIRBuilder):
    @builder.func([(2, 3, 64, 32), (2, 3, 64, 32)], [torch.float32, torch.float32])
    def elementwise_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.add(in0, in1)


def module_elementwise_multiply_3d_multiply(builder: TTIRBuilder):
    @builder.func([(3, 32, 64), (3, 32, 64)], [torch.float32, torch.float32])
    def elementwise_multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.multiply(in0, in1)


@pytest.mark.parametrize(
    "test_func,test_name",
    [
        pytest.param(module_elementwise_add_3d_add, "3d_add", id="3d_add"),
        pytest.param(
            module_elementwise_multiply_3d_multiply,
            "3d_multiply",
            id="3d_multiply",
        ),
        pytest.param(module_elementwise_add_4d_add, "4d_add", id="4d_add"),
    ],
)
@pytest.mark.parametrize(
    "collapse_tensors", [True, False], ids=["collapsed", "non_collapsed"]
)
@pytest.mark.parametrize("target", ["ttmetal"], ids=["ttmetal"])
def test_binary_ops_collapse_tensors(
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


@pytest.mark.parametrize("target", ["ttmetal"])
def test_binary_ops_auto_reblock_large_tensor(request, device, target: str):
    shape = (1024, 1024)

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.float32, torch.float32])
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            return add(in0, in1, builder)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        print_ir=False,
    )


# Logical binary ops with custom golden tensors containing mix of 0s and non-0s
logical_ops = [
    logical_and,
    logical_or,
    logical_xor,
]

logical_ops_dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.int32 | SkipIf("sim"),
    torch.int64 | SkipIf("sim"),
    torch.bool | SkipIf("sim"),
]


def create_logical_op_goldens(
    shape: Shape, dtype: torch.dtype, op_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create golden tensors with a mix of 0s and non-0s for logical op testing."""
    golden0 = torch.zeros(shape, dtype=dtype)
    golden1 = torch.zeros(shape, dtype=dtype)

    golden0[: shape[0] // 2, :] = torch.randn(shape[0] // 2, shape[1])
    golden1[:, ::2] = torch.randn(shape[0], shape[1] // 2)

    bool0 = golden0 != 0
    bool1 = golden1 != 0
    if op_name == "logical_and":
        output_golden = (bool0 & bool1).to(dtype)
    elif op_name == "logical_or":
        output_golden = (bool0 | bool1).to(dtype)
    elif op_name == "logical_xor":
        output_golden = (bool0 ^ bool1).to(dtype)
    else:
        raise ValueError(f"Unknown logical op: {op_name}")

    return golden0, golden1, output_golden


@pytest.mark.parametrize(
    "shape,dtype,test_fn",
    rotated_params(
        ELEMENTWISE_2D_SHAPES, logical_ops, logical_ops_dtypes, value_order=[0, 2, 1]
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_logical_ops(
    shape: Shape, dtype: torch.dtype, test_fn: Callable, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def logical_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            golden0, golden1, output_golden = create_logical_op_goldens(
                shape, dtype, test_fn.__name__
            )
            result = test_fn(in0, in1, builder)
            builder.set_goldens(
                inputs={in0: golden0, in1: golden1}, outputs={result: output_golden}
            )
            return result

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=[],
    )


# Scalar binary ops
def add_scalar(
    in0: Operand,
    scalar_value: float,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value, dtype=dtype))
    return builder.add(in0, scalar, unit_attrs=unit_attrs)


def multiply_scalar(
    in0: Operand,
    scalar_value: float,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value, dtype=dtype))
    return builder.multiply(in0, scalar, unit_attrs=unit_attrs)


def subtract_scalar(
    in0: Operand,
    scalar_value: float,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value, dtype=dtype))
    return builder.subtract(in0, scalar, unit_attrs=unit_attrs)


def div_scalar(
    in0: Operand,
    scalar_value: float,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value, dtype=dtype))
    return builder.div(in0, scalar, unit_attrs=unit_attrs)


def pow_scalar(
    in0: Operand,
    scalar_value: float,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value, dtype=dtype))
    return builder.pow(in0, scalar, unit_attrs=unit_attrs)


scalar_binary_ops = [
    pytest.param(add_scalar, 2.5, id="add_2p5"),
    pytest.param(add_scalar, 5, id="add_5"),
    pytest.param(multiply_scalar, 3.7, id="multiply_3p7"),
    pytest.param(subtract_scalar, 1.5, id="subtract_1p5"),
    pytest.param(subtract_scalar, 3, id="subtract_3"),
    pytest.param(div_scalar, 2.5, id="div_2p5"),
    pytest.param(pow_scalar, 2.0, id="pow_2p0"),
]

scalar_binary_dtypes = [
    torch.float32,
    torch.bfloat16,
]


scalar_binary_int_ops = [
    pytest.param(add_scalar, 5, id="add_5"),
    pytest.param(subtract_scalar, 3, id="subtract_3"),
]


def _compile_scalar_binary_op(
    test_fn: Callable,
    scalar_value: float,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    int_dtypes = (torch.int32, torch.int64)
    float_only_ops = ("multiply_scalar", "div_scalar", "pow_scalar")
    is_int = dtype in int_dtypes
    is_fractional = scalar_value != int(scalar_value)

    if is_int and test_fn.__name__ in float_only_ops:
        pytest.skip(f"{test_fn.__name__} not supported for {dtype}")
    if is_int and is_fractional:
        pytest.skip(f"fractional scalar {scalar_value} not valid for {dtype}")

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def scalar_op_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return test_fn(in0, scalar_value, dtype, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


_SCALAR_BINARY_PARAMS = rotated_params(
    scalar_binary_ops,
    ELEMENTWISE_2D_SHAPES,
    scalar_binary_dtypes,
    value_order=[2, 3, 0, 1],
) + rotated_params(
    ELEMENTWISE_2D_SHAPES,
    [torch.int32 | SkipIf("sim")],
    scalar_binary_int_ops,
)


@pytest.mark.parametrize("shape,dtype,test_fn,scalar_value", _SCALAR_BINARY_PARAMS)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_scalar_binary_ops(
    shape: Shape,
    dtype: torch.dtype,
    test_fn: Callable,
    scalar_value: float,
    target: str,
    request,
    device,
):
    """Test binary operations with scalar operands across dtypes on ttmetal."""
    _compile_scalar_binary_op(
        test_fn, scalar_value, shape, dtype, target, request, device
    )


# Scalar comparison ops
SCALAR_CMP_VALUE = 5


def _make_scalar_cmp_fn(op_name):
    def fn(in0, scalar_value, dtype, builder, unit_attrs=None):
        shape = builder.get_shape(in0)
        scalar = builder.constant(torch.full(shape, scalar_value, dtype=dtype))
        return getattr(builder, op_name)(in0, scalar, unit_attrs=unit_attrs)

    fn.__name__ = f"{op_name}_scalar"
    return fn


scalar_comparison_ops = [
    _make_scalar_cmp_fn("eq"),
    _make_scalar_cmp_fn("ne"),
    _make_scalar_cmp_fn("gt"),
    _make_scalar_cmp_fn("ge"),
    _make_scalar_cmp_fn("lt"),
    _make_scalar_cmp_fn("le"),
]

scalar_comparison_dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.int32 | SkipIf("sim"),
]


def _make_scalar_cmp_golden(shape, dtype, op_name):
    n = 1
    for d in shape:
        n *= d
    sv = SCALAR_CMP_VALUE

    if op_name in ("eq_scalar", "ne_scalar"):
        t = torch.randint(sv - 10, sv + 10, (n,)).to(torch.float32)
        t[torch.randperm(n)[: n // 2]] = float(sv)
        return t.reshape(shape).to(dtype)

    if dtype in (torch.int32, torch.int64):
        return torch.randint(sv - 10, sv + 11, (n,)).reshape(shape).to(dtype)
    return (torch.rand(n) * 20 + (sv - 10)).reshape(shape).to(dtype)


@pytest.mark.parametrize(
    "shape,dtype,test_fn",
    rotated_params(
        scalar_comparison_ops,
        ELEMENTWISE_2D_SHAPES,
        scalar_comparison_dtypes,
        value_order=[1, 2, 0],
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_scalar_comparison_ops(
    shape: Shape,
    dtype: torch.dtype,
    test_fn: Callable,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def scalar_cmp_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            golden = _make_scalar_cmp_golden(shape, dtype, test_fn.__name__)
            builder.set_goldens(inputs={in0: golden})
            return test_fn(in0, SCALAR_CMP_VALUE, dtype, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Binary bitwise ops (int only)
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


binary_bitwise_ops = [
    bitwise_and,
    bitwise_or,
    bitwise_xor,
]

binary_bitwise_dtypes = [
    torch.int32 | SkipIf("sim"),
    torch.uint32 | SkipIf("sim"),
    torch.uint16 | SkipIf("sim"),
    torch.uint8 | SkipIf("sim"),
]


@pytest.mark.parametrize(
    "shape,dtype,test_fn",
    rotated_params(
        ELEMENTWISE_2D_SHAPES,
        binary_bitwise_ops,
        binary_bitwise_dtypes,
        value_order=[0, 2, 1],
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_bitwise_binary_ops(
    shape: Shape, dtype: torch.dtype, test_fn: Callable, target: str, request, device
):
    if dtype == torch.uint8:
        pytest.xfail("uint8 bitwise ops are not supported on ttmetal yet")

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            return test_fn(in0, in1, builder)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Binary logical shift ops (int only)
def logical_left_shift(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_left_shift(in0, in1, unit_attrs=unit_attrs)


def logical_right_shift(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_right_shift(in0, in1, unit_attrs=unit_attrs)


def right_shift(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.right_shift(in0, in1, unit_attrs=unit_attrs)


binary_logical_shift_ops = [
    logical_left_shift,
    logical_right_shift,
]

binary_logical_shift_dtypes = [
    torch.int32 | SkipIf("sim"),
    torch.uint32 | SkipIf("sim"),
    torch.uint16 | SkipIf("sim"),
]


@pytest.mark.parametrize(
    "shape,dtype,test_fn",
    rotated_params(
        ELEMENTWISE_2D_SHAPES,
        binary_logical_shift_ops,
        binary_logical_shift_dtypes,
        value_order=[0, 2, 1],
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_logical_shift_binary_ops(
    shape: Shape, dtype: torch.dtype, test_fn: Callable, target: str, request, device
):
    if test_fn == logical_left_shift and dtype == torch.uint16:
        pytest.xfail("uint16 logical left shift op is not supported yet")

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            return test_fn(in0, in1, builder)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


binary_right_shift_dtypes = [
    torch.int32 | SkipIf("sim"),
]


@pytest.mark.parametrize("shape", ELEMENTWISE_2D_SHAPES, ids=shape_str)
@pytest.mark.parametrize("dtype", binary_right_shift_dtypes, ids=["i32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_right_shift_binary_op(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            lhs = torch.randint(-(2**31), 2**31, shape, dtype=dtype)
            rhs = torch.randint(0, 32, shape, dtype=dtype)
            result = right_shift(in0, in1, builder)
            output = torch.bitwise_right_shift(lhs, rhs).to(dtype)
            builder.set_goldens(inputs={in0: lhs, in1: rhs}, outputs={result: output})
            return result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


def _make_shift_edge_case_tensors(shape: Shape) -> Tuple[torch.Tensor, torch.Tensor]:
    # Include sign-extreme and mixed-sign values in lhs, and boundary shifts in rhs.
    lhs_pattern = torch.tensor(
        [-(2**31), -1024, -17, -1, 0, 1, 17, 1024, 2**31 - 1], dtype=torch.int32
    )
    rhs_pattern = torch.tensor([0, 1, 2, 7, 15, 16, 30, 31, 0], dtype=torch.int32)

    lhs_flat = lhs_pattern.repeat(
        (shape[0] * shape[1] + len(lhs_pattern) - 1) // len(lhs_pattern)
    )[: shape[0] * shape[1]]
    rhs_flat = rhs_pattern.repeat(
        (shape[0] * shape[1] + len(rhs_pattern) - 1) // len(rhs_pattern)
    )[: shape[0] * shape[1]]
    return lhs_flat.reshape(shape), rhs_flat.reshape(shape)


def _logical_left_shift_edge_golden(
    lhs: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    lhs_i64 = lhs.to(torch.int64)
    rhs_i64 = rhs.to(torch.int64)
    lhs_unsigned = torch.bitwise_and(lhs_i64, 0xFFFFFFFF)
    out = torch.bitwise_left_shift(lhs_unsigned, rhs_i64)
    return torch.bitwise_and(out, 0xFFFFFFFF).to(torch.int32)


def _logical_right_shift_edge_golden(
    lhs: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    lhs_i64 = lhs.to(torch.int64)
    rhs_i64 = rhs.to(torch.int64)
    lhs_unsigned = torch.bitwise_and(lhs_i64, 0xFFFFFFFF)
    out = torch.bitwise_right_shift(lhs_unsigned, rhs_i64)
    return torch.bitwise_and(out, 0xFFFFFFFF).to(torch.int32)


@pytest.mark.parametrize(
    "shape,op_name",
    rotated_params(
        ELEMENTWISE_2D_SHAPES,
        ["logical_left_shift", "logical_right_shift", "right_shift"],
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.skip_config(["sim"])
def test_shift_binary_ops_edge_cases(
    shape: Shape, op_name: str, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.int32, torch.int32])
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            lhs, rhs = _make_shift_edge_case_tensors(shape)

            if op_name == "logical_left_shift":
                result = logical_left_shift(in0, in1, builder)
                output = _logical_left_shift_edge_golden(lhs, rhs)
            elif op_name == "logical_right_shift":
                result = logical_right_shift(in0, in1, builder)
                output = _logical_right_shift_edge_golden(lhs, rhs)
            else:
                result = right_shift(in0, in1, builder)
                output = torch.bitwise_right_shift(lhs, rhs).to(torch.int32)

            builder.set_goldens(inputs={in0: lhs, in1: rhs}, outputs={result: output})
            return result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Binary comparison ops
def eq(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.eq(in0, in1, unit_attrs=unit_attrs)


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


def ne(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.ne(in0, in1, unit_attrs=unit_attrs)


binary_comparison_ops = [
    eq,
    ge,
    gt,
    le,
    lt,
    ne,
]

binary_comparison_cases = binary_comparison_ops + [ne]

binary_comparison_dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.int32 | SkipIf("sim"),
    torch.int64 | SkipIf("sim"),
    torch.bool | SkipIf("sim"),
    torch.float32,
    torch.uint8 | SkipIf("ttmetal"),
]


@pytest.mark.parametrize(
    "shape,dtype,test_fn",
    rotated_params(
        binary_comparison_cases,
        ELEMENTWISE_2D_SHAPES,
        binary_comparison_dtypes,
        value_order=[1, 2, 0],
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_comparison_ops(
    shape: Shape,
    dtype: torch.dtype,
    test_fn: Callable,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def comparison_ops(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            randn_tensor1 = torch.randn(shape, dtype=torch.float32)
            randn_tensor2 = torch.randn(shape, dtype=torch.float32)

            num_elements = torch.numel(randn_tensor1)
            num_equal_indices = num_elements // 2

            equal_indices = torch.randperm(num_elements)[:num_equal_indices]
            randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[
                equal_indices
            ]

            input_tensor1 = randn_tensor1.to(dtype)
            input_tensor2 = randn_tensor2.to(dtype)

            builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

            return test_fn(in0, in1, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


implicit_bcast_inner_2D_shapes = [
    (32, 32),
    (32, 96),
    (96, 32),
    (96, 96),
    (416, 32),
    (32, 416),
    (416, 96),
    (96, 416),
    (416, 416),
]


@pytest.mark.parametrize(
    "shape,dtype",
    rotated_params(implicit_bcast_inner_2D_shapes, [torch.float32, torch.bfloat16]),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_implicit_bcast_inner_2D(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    shape_F = shape
    shape_C = (shape_F[0], 1)
    shape_R = (1, shape_F[1])
    shape_S = (1, 1)

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape_C, shape_R, shape_S], [dtype, dtype, dtype, dtype])
        def bcast_all_cases(
            in_F0: Operand,
            in_C0: Operand,
            in_R0: Operand,
            in_S0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            tensor_F0 = torch.rand(shape_F, dtype=dtype) * 2.0 - 1.0
            tensor_C0 = torch.rand(shape_C, dtype=dtype) * 2.0 - 1.0
            tensor_R0 = torch.rand(shape_R, dtype=dtype) * 2.0 - 1.0
            tensor_S0 = torch.rand(shape_S, dtype=dtype) - 0.5

            builder.set_goldens(
                inputs={
                    in_F0: tensor_F0,
                    in_C0: tensor_C0,
                    in_R0: tensor_R0,
                    in_S0: tensor_S0,
                }
            )

            in_R1 = builder.add(
                builder.add(in_S0, in_R0, unit_attrs=unit_attrs),
                in_S0,
                unit_attrs=unit_attrs,
            )
            in_C1 = builder.subtract(
                in_S0,
                builder.subtract(in_C0, in_S0, unit_attrs=unit_attrs),
                unit_attrs=unit_attrs,
            )
            in_F1 = builder.add(in_C1, in_R1, unit_attrs=unit_attrs)
            return builder.add(in_F1, in_S0, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


binary_broadcast_shard_dim_shapes = [
    [(1, 2, 1, 32), (1, 1, 1, 32)],
    [(1, 16, 1, 32), (1, 1, 1, 32)],
    [(1, 1, 1, 32), (1, 2, 1, 32)],  # broadcast dim1
    [(2, 2, 1, 32), (1, 2, 1, 32)],  # broadcast dim0
    # 3D shape
    [(3, 16, 32), (1, 16, 32)],
    # 5D shape
    [(1, 1, 1, 32, 32), (1, 1, 8, 32, 32)],
    # Larger tensors
    [(1, 2, 64, 64), (1, 1, 64, 64)],
    [(1, 4, 64, 128), (1, 1, 64, 128)],
    [(1, 1, 8, 64, 64), (1, 1, 1, 64, 64)],
    # Broadcast on row/col dims
    [(1, 2, 32, 32), (1, 2, 1, 32)],
    [(1, 4, 64, 128), (1, 4, 1, 128)],
    [(1, 1, 32, 32), (1, 1, 32, 1)],
    # Broadcast on all dims
    [(19, 160, 64), (1, 1, 1)],
]


@pytest.mark.parametrize(
    "shapes,dtype,test_fn",
    rotated_params(
        binary_broadcast_shard_dim_shapes,
        [torch.float32],
        [add, subtract, multiply],
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_binary_ops_broadcast_shard_dims(
    shapes: List[Shape],
    dtype: torch.dtype,
    test_fn: Callable,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype, dtype])
        def binary_broadcast(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return test_fn(in0, in1, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        print_ir=False,
        device=device,
    )


# ============================================================
# Tests moved from test_ttir_ops.py during TTMetal test
# reorganization.
# ============================================================


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_div(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.float32, torch.float32])
        def div(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            dividend_tensor = builder._get_golden_tensor(in0)
            divisor_tensor = builder._get_golden_tensor(in1)

            dividend_tensor = dividend_tensor.apply_shardwise(
                lambda shard: (
                    shard.__setitem__(shard.abs() < 0.01, 0.03) or shard
                    if torch.is_floating_point(shard)
                    else shard
                )
            )

            divisor_tensor = divisor_tensor.apply_shardwise(
                lambda shard: (
                    shard.__setitem__(shard.abs() < 0.01, -0.03) or shard
                    if torch.is_floating_point(shard)
                    else shard
                )
            )

            output_golden = torch.div(dividend_tensor, divisor_tensor)
            div0 = builder.div(in0, in1, unit_attrs=unit_attrs)
            builder.set_goldens_from_builder_tensor(
                {in0: dividend_tensor, in1: divisor_tensor}, {div0: output_golden}
            )
            return div0

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shape",
    UNALIGNED_SHAPES
    + [
        pytest.param(
            (677, 1, 1), marks=pytest.mark.skip_config(["n150"])
        ),  # TODO (anuragsingh): Fix nondeterministic issue with Allocator for this test.
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unaligned_shapes_add(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def add(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Magnitudes of the elements should be in [0.01, 1) to avoid FP accuracy issue.
            tensor_lhs = torch.rand(shape, dtype=dtype) * 0.99 + 0.01
            tensor_rhs = torch.rand(shape, dtype=dtype) * 0.99 + 0.01
            signs_lhs = torch.randint(0, 2, shape) * 2 - 1
            signs_rhs = torch.randint(0, 2, shape) * 2 - 1
            tensor_lhs *= signs_lhs
            tensor_rhs *= signs_rhs
            builder.set_goldens(inputs={in0: tensor_lhs, in1: tensor_rhs})
            return builder.add(in0, in1)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
