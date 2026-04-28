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
    SkipIf,
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
    atan2,
    div,
    gelu_backward,
    gelu_backward_tanh,
    maximum | SkipIf("sim"),
    minimum | SkipIf("sim"),
    multiply,
    pow,
    remainder,
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
@pytest.mark.parametrize("target", ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")])
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
    if dtype == torch.int32 or dtype == torch.int64:
        pytest.skip("unsupported/not guaranteed to work")

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
        print_ir=False,
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
@pytest.mark.parametrize("target", ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")])
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
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize("test_fn", binary_bitwise_ops)
def test_bitwise_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
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
    torch.int32 | SkipIf("sim"),
    torch.uint32 | SkipIf("sim"),
    torch.uint16 | SkipIf("sim"),
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
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.int32 | SkipIf("sim"),
        torch.int64 | SkipIf("sim"),
        torch.bool | SkipIf("sim"),
        torch.uint8,
    ],
    ids=["f32", "bf16", "i32", "i64", "i1", "u8"],
)
@pytest.mark.parametrize("target", ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")])
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


# CPU-hoisted binary ops.
hoisted_binary_ops_float = [
    atan2,
    gelu_backward,
    gelu_backward_tanh,
]

hoisted_binary_ops_float_integer = [
    add,
    div,
    maximum,
    minimum,
    multiply,
    eq,
    ne,
    gt,
    ge,
    lt,
    le,
    subtract,
    remainder,
]

hoisted_binary_ops_integer = [
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    # logical_left_shift and logical_right_shift are excluded because random i32
    # shift amounts (values >= 32 or negative) produce all-zero outputs, making
    # PCC comparison degenerate. They are tested separately in
    # test_hoisted_logical_shift_ops with controlled shift amounts.
]

hoisted_binary_shapes = [
    [(128, 128), (128, 128)],  # Same shapes
    [(128, 128), (1, 128)],  # Broadcasting second dimension
    [(128, 128), (128, 1)],  # Broadcasting first dimension
    [(1, 32), (1, 32)],  # Small shapes
    [(1, 32), (32, 1)],  # Both operands broadcast
    [(1, 1), (32, 32)],  # Scalar-like broadcast
    [(128, 128, 64), (128, 1, 64)],  # 3D tensors with broadcasting
    [(1, 1, 64), (128, 128, 64)],  # 3D both leading dims broadcast
    [(7, 41, 43, 11), (7, 41, 43, 11)],  # 4D tensors
]


@x86_only
@pytest.mark.parametrize("shapes", hoisted_binary_shapes, ids=shapes_list_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("test_fn", hoisted_binary_ops_float)
@pytest.mark.parametrize(
    "target",
    ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")],
)
def test_cpu_hoistable_binary_ops_float(
    test_fn: Callable,
    shapes: List[Shape],
    dtype: torch.dtype,
    request,
    target: str,
    device,
):
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
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize("shapes", hoisted_binary_shapes, ids=shapes_list_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("test_fn", hoisted_binary_ops_float_integer)
@pytest.mark.parametrize(
    "target",
    ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")],
)
def test_cpu_hoistable_binary_ops_float_integer(
    test_fn: Callable,
    shapes: List[Shape],
    dtype: torch.dtype,
    request,
    target: str,
    device,
):
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
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize(
    "target",
    ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")],
)
@pytest.mark.parametrize("test_fn", logical_ops)
def test_hoisted_logical_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    # Separate from the generic hoisted test because logical ops need custom
    # goldens with a deliberate mix of zeros and non-zeros to exercise the
    # logical truth table. Random float inputs would rarely produce exact zeros.
    # These ops are excluded from hoisted_binary_ops_float to avoid duplication.
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


@x86_only
@pytest.mark.parametrize("shapes", hoisted_binary_shapes, ids=shapes_list_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize("test_fn", hoisted_binary_ops_integer)
@pytest.mark.parametrize(
    "target",
    ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")],
)
def test_cpu_hoistable_binary_ops_integer(
    test_fn: Callable,
    shapes: List[Shape],
    dtype: torch.dtype,
    request,
    target: str,
    device,
):
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
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize("test_fn", binary_logical_shift_ops)
@pytest.mark.parametrize(
    "target",
    ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")],
)
def test_hoisted_logical_shift_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    # Separate from the generic hoisted integer test because random shift
    # amounts (values >= 32 or negative) produce all-zero outputs, making PCC
    # comparison degenerate. We use controlled shift amounts in [0, 31].
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def hoisted_shift_op_fn(
            in0: Operand, in1: Operand, builder: TTIRBuilder
        ) -> Operand:
            shift_amounts = torch.randint(0, 32, shape, dtype=dtype)
            builder.set_goldens(inputs={in1: shift_amounts})
            return test_fn(in0, in1, builder, unit_attrs=["ttir.should_hoist"])

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
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int32 | SkipIf("sim")], ids=["f32", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn" | SkipIf("sim")])
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
@pytest.mark.parametrize(
    "target",
    ["ttnn" | SkipIf("sim"), "emitpy" | SkipIf("sim")],
)
def test_hoisted_pow(shape: Shape, dtype: torch.dtype, target: str, request, device):
    # Separate from the generic hoisted test because pow needs torch.abs() on
    # the base operand to avoid negative bases with fractional exponents (NaN).
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
