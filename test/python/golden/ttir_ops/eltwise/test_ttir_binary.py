# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from golden import get_golden_function
from builder.base.builder_apis import (
    compile_and_execute_ttir,
)
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
)
from ttmlir.dialects import ttir


pytestmark = pytest.mark.frontend("ttir")


# Binary ops
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
    return builder.maximum(in0, in1)


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
    maximum,
    minimum,
    multiply,
    pow,
    remainder | Marks(pytest.mark.skip_config(["ttmetal"])),
    subtract,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
@pytest.mark.parametrize("test_fn", binary_ops)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    # FP32 pow fails due to tt-metal untilize NaN handling.
    # See: https://github.com/tenstorrent/tt-metal/pull/33904
    if test_fn.__name__ == "pow" and dtype == torch.float32 and target == "ttnn":
        pytest.xfail(
            "FP32 pow fails due to tt-metal untilize NaN handling. "
            "See: https://github.com/tenstorrent/tt-metal/pull/33904"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            return test_fn(in0, in1, builder)

    pipeline_options = []
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
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
    """Create golden tensors with a mix of 0s and non-0s for logical op testing.

    Returns:
        Tuple of (golden0, golden1, output_golden) tensors.
    """
    # Pattern: alternating rows of zeros and non-zeros for variety.
    golden0 = torch.zeros(shape, dtype=dtype)
    golden1 = torch.zeros(shape, dtype=dtype)

    # in0: first half rows are non-zero, second half are zero
    golden0[: shape[0] // 2, :] = torch.randn(shape[0] // 2, shape[1])
    # in1: alternating columns of zero and non-zero
    golden1[:, ::2] = torch.randn(shape[0], shape[1] // 2)

    # Compute expected output based on logical operation.
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
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
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

    pipeline_options = []
    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize("test_fn", logical_ops)
def test_hoisted_logical_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def hoisted_logical_op_fn(
            in0: Operand, in1: Operand, builder: TTIRBuilder
        ) -> Operand:
            golden0, golden1, output_golden = create_logical_op_goldens(
                shape, dtype, test_fn.__name__
            )
            result = test_fn(in0, in1, builder, unit_attrs=["ttir.should_hoist"])
            builder.set_goldens(
                inputs={in0: golden0, in1: golden1}, outputs={result: output_golden}
            )
            return result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Scalar binary ops
def add_scalar(
    in0: Operand,
    scalar_value: float,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    """Add a scalar value to a tensor"""
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value))
    return builder.add(in0, scalar, unit_attrs=unit_attrs)


def multiply_scalar(
    in0: Operand,
    scalar_value: float,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    """Multiply a tensor by a scalar value"""
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value))
    return builder.multiply(in0, scalar, unit_attrs=unit_attrs)


def subtract_scalar(
    in0: Operand,
    scalar_value: float,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    """Subtract a scalar value from a tensor"""
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value))
    return builder.subtract(in0, scalar, unit_attrs=unit_attrs)


def div_scalar(
    in0: Operand,
    scalar_value: float,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    """Divide a tensor by a scalar value"""
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value))
    return builder.div(in0, scalar, unit_attrs=unit_attrs)


def pow_scalar(
    in0: Operand,
    scalar_value: float,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    """Raise a tensor to a scalar power"""
    shape = builder.get_shape(in0)
    scalar = builder.constant(torch.full(shape, scalar_value))
    return builder.pow(in0, scalar, unit_attrs=unit_attrs)


scalar_binary_ops = [
    (add_scalar, 2.5),
    (multiply_scalar, 3.0),
    (subtract_scalar, 1.5),
    (div_scalar, 3.0)
    | Marks(
        pytest.mark.xfail(
            reason="Fails atol and rtol, issue here: https://github.com/tenstorrent/tt-mlir/issues/5924"
        )
    ),
    (pow_scalar, 2.0),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "test_fn,scalar_value",
    scalar_binary_ops,
    ids=[
        "add_scalar",
        "multiply_scalar",
        "subtract_scalar",
        "div_scalar",
        "pow_scalar",
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
    """Test binary operations with scalar operands on ttmetal"""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def scalar_op_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return test_fn(in0, scalar_value, builder, unit_attrs=unit_attrs)

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
    torch.int32,
    torch.uint32,
    torch.uint16,
    torch.uint8,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", binary_bitwise_dtypes, ids=["i32", "u32", "u16", "u8"]
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
@pytest.mark.parametrize("test_fn", binary_bitwise_ops)
def test_bitwise_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if target == "emitpy" and (dtype == torch.uint16 or dtype == torch.uint32):
        pytest.xfail("uint16 and uint32 aren't supported in ttnn pybinds")
    elif target == "ttmetal":
        pytest.xfail(
            "ttmetal does not support bitwise ops for integers due to tilize/untilize."
        )

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


binary_logical_shift_ops = [
    logical_left_shift,
    logical_right_shift,
]


binary_logical_shift_dtypes = [
    torch.int32,
    torch.uint32,
    torch.uint16,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", binary_logical_shift_dtypes, ids=["i32", "u32", "u16"]
)
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize("test_fn", binary_logical_shift_ops)
def test_logical_shift_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if test_fn == logical_left_shift and dtype == torch.uint16:
        pytest.xfail("uint16 logical left shift op is not supported yet")
    if target == "emitpy" and (dtype == torch.uint16 or dtype == torch.uint32):
        pytest.xfail("uint16 and uint32 aren't supported in ttnn pybinds")

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
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
@pytest.mark.parametrize("test_fn", binary_comparison_ops)
def test_comparison_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    if target == "ttmetal" and dtype == torch.int32:
        pytest.skip("ttmetal does not support int32 comparison ops")

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

            # Set some indices in randn_tensor2 to be the same as randn_tensor1
            # This ensures we have both equal and unequal values for comprehensive testing
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


# Unaligned shapes for the add op
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
    (51, 19, 23),
    (677, 1, 1),
    (2, 3, 5, 7),
    (3, 37, 5, 53),
    (37, 3, 5, 53),
    (41, 7, 43, 11),
    (7, 41, 43, 11),
    (1, 23, 1, 1),
    (23, 1, 1, 1),
    (3, 5, 7, 11, 13),
]


@pytest.mark.parametrize("shape", unaligned_shapes, ids=shape_str)
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


# Hoisted binary ops
def create_hoisted_binary_op(op_func, name):
    """Create a hoisted version of a binary operation by adding the should_hoist unit attribute"""

    def hoisted_op(in0, in1, builder, **kwargs):
        return op_func(in0, in1, builder, unit_attrs=["ttir.should_hoist"], **kwargs)

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


hoisted_binary_ops = [
    add,
    multiply,
    subtract,
    # TODO(#6183): Re-enable when F32 untilize on-device precision loss is fixed
    # F32 untilize on-device introduces precision loss that causes close values to become
    # identical, breaking exact equality comparisons in CPU-hoisted ops.
    # See: https://github.com/tenstorrent/tt-mlir/issues/6183
    # eq,
    ne,
    gt,
    ge,
    lt,
    le,
    minimum,
    maximum,
]


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        [(128, 128), (128, 128)],  # Same shapes
        [(128, 128), (1, 128)],  # Broadcasting second dimension
        [(128, 128), (128, 1)],  # Broadcasting first dimension
        [(128, 128, 64), (128, 1, 64)],  # 3D tensors with broadcasting
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("test_fn", hoisted_binary_ops)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_binary_ops(
    test_fn: Callable,
    shapes: List[Shape],
    dtype: torch.dtype,
    request,
    target: str,
    device,
):
    """Test binary ops that support CPU hoisting"""

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def hoisted_binary_op_fn(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ) -> Operand:
            return test_fn(in0, in1, builder, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=f"{request.node.name}",
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


@pytest.mark.skip_config(["p150"], ["p300"], reason="See issue #6565")
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

    # Avoid too many test entries, test LHS & RHS bcast together.
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
        [(1, 16, 32), (1, 16, 32)],
        # 5D shape
        [(1, 1, 1, 32, 32), (1, 1, 8, 32, 32)],
        # Larger tensors
        [(1, 2, 64, 64), (1, 1, 64, 64)],
        [(1, 4, 64, 128), (1, 1, 64, 128)],
        [(1, 1, 8, 64, 64), (1, 1, 1, 64, 64)],
        # broadcast on row/col dims
        [(1, 2, 32, 32), (1, 2, 1, 32)],
        [(1, 4, 64, 128), (1, 4, 1, 128)],
        [(1, 1, 32, 32), (1, 1, 32, 1)],
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
        device=device,
    )


# Binary eltwise ops with implicit broadcasting
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
        add,
        multiply,
        subtract,
        eq,
        ne,
        le,
        lt,
        ge,
        gt,
        div,
        remainder,
        maximum,
        minimum,
        pow | Marks(pytest.mark.xfail(reason="Golden Failure")),
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
    device,
):
    # FP32 eq broadcast fails due to tt-metal untilize NaN handling.
    # See: https://github.com/tenstorrent/tt-metal/pull/33904
    if test_fn == eq and dtype == torch.float32 and target == "ttnn":
        pytest.xfail(
            "FP32 eq broadcast fails due to tt-metal untilize NaN handling. "
            "See: https://github.com/tenstorrent/tt-metal/pull/33904"
        )

    pcc = 0.99

    if test_fn == div:
        pcc = 0.97

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype, dtype])
        def binary_eltwise_op_fn(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ) -> Operand:
            return test_fn(in0, in1, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pcc=pcc,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_pow(shape: Shape, dtype: torch.dtype, target: str, request, device):
    # FP32 pow fails due to tt-metal untilize NaN handling.
    # See: https://github.com/tenstorrent/tt-metal/pull/33904
    if dtype == torch.float32 and target == "ttnn":
        pytest.xfail(
            "FP32 hoisted_pow fails due to tt-metal untilize NaN handling. "
            "See: https://github.com/tenstorrent/tt-metal/pull/33904"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def hoisted_pow_wrapper(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return pow(in0, in1, builder, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
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
