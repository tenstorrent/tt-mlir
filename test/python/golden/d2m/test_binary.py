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
from test_utils import Marks, shape_str, shapes_list_str, SkipIf

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


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.int32 | SkipIf("sim"),
        torch.int64 | SkipIf("sim"),
    ],
    ids=["f32", "bf16", "i32", "i64"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("test_fn", binary_ops)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
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


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.int32 | SkipIf("sim"),
        torch.int64 | SkipIf("sim"),
        torch.bool | SkipIf("sim"),
    ],
    ids=["f32", "bf16", "i32", "i64", "i1"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("test_fn", logical_ops)
def test_logical_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
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
    (add_scalar, 2.5),
    (add_scalar, 5),
    (multiply_scalar, 3.7),
    (subtract_scalar, 1.5),
    (subtract_scalar, 3),
    (div_scalar, 2.5),
    (pow_scalar, 2.0),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.int32 | SkipIf("sim"),
    ],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "test_fn,scalar_value",
    scalar_binary_ops,
    ids=[
        "add_2.5",
        "add_5",
        "multiply_3.7",
        "subtract_1.5",
        "subtract_3",
        "div_2.5",
        "pow_2.0",
    ],
)
def test_scalar_binary_ops(
    test_fn: Callable,
    scalar_value: float,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Test binary operations with scalar operands across f32, bf16, and i32 on ttmetal"""
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


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.int32 | SkipIf("sim"),
    ],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("test_fn", scalar_comparison_ops)
def test_scalar_comparison_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
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


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", binary_bitwise_dtypes, ids=["i32", "u32", "u16", "u8"]
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("test_fn", binary_bitwise_ops)
def test_bitwise_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
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


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.int32 | SkipIf("sim"),
        torch.int64 | SkipIf("sim"),
        torch.bool | SkipIf("sim"),
        torch.uint8 | SkipIf("ttmetal"),
    ],
    ids=["f32", "bf16", "i32", "i64", "i1", "u8"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("test_fn", binary_comparison_ops)
def test_comparison_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
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


@pytest.mark.parametrize("shape", implicit_bcast_inner_2D_shapes, ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
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


@pytest.mark.parametrize(
    "shapes",
    [
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
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [add, subtract, multiply],
    ids=["add", "subtract", "multiply"],
)
def test_binary_ops_broadcast_shard_dims(
    test_fn: Callable,
    shapes: List[Shape],
    dtype: torch.dtype,
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


# 1D tensor test for ttmetal
@pytest.mark.parametrize("shape", [(128,)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_1d(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def binary_1d(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return add(in0, in1, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
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
