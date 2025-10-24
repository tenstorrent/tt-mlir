# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base import get_golden_function
from builder.base.builder_utils import (
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


binary_ops = [
    add,
    atan2 | Marks(pytest.mark.skip_config(["ttmetal"])),
    div,
    logical_and | Marks(pytest.mark.skip_config(["ttmetal"])),
    logical_or | Marks(pytest.mark.skip_config(["ttmetal"])),
    logical_xor | Marks(pytest.mark.skip_config(["ttmetal"])),
    maximum
    | Marks(
        pytest.mark.skip_config(
            ["ttmetal"], reason="https://github.com/tenstorrent/tt-mlir/issues/5016"
        )
    ),
    minimum | Marks(pytest.mark.skip_config(["ttmetal"])),
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
    pipeline_options = []
    compile_and_execute_ttir(
        test_fn,
        inputs_shapes=[shape, shape],
        inputs_types=[dtype, dtype],
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
    torch.int32,
    torch.uint32,
    torch.uint16,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", binary_bitwise_dtypes, ids=["i32", "u32", "u16"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize("test_fn", binary_bitwise_ops)
def test_bitwise_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_ttir(
        test_fn,
        inputs_shapes=[shape, shape],
        inputs_types=[dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
    logical_left_shift_0 = builder.logical_left_shift(in0, in1, unit_attrs=unit_attrs)
    # Constrain shift amounts to be within valid range
    shift_tensor_1 = builder._get_golden_tensor(in1)
    dtype_bits = torch.iinfo(shift_tensor_1.shard_at(0).dtype).bits
    # Handle uint32 which doesn't support % operator in PyTorch
    constrained_shift_tensor = shift_tensor_1.apply_shardwise(
        lambda shard: (shard.to(torch.int64) % dtype_bits).to(shard.dtype)
    )

    golden_fn = get_golden_function(ttir.LogicalLeftShiftOp)
    output_golden = golden_fn(builder._get_golden_tensor(in0), constrained_shift_tensor)
    builder.set_goldens_from_builder_tensor(
        {in1: constrained_shift_tensor}, {logical_left_shift_0: output_golden}
    )
    return logical_left_shift_0


def logical_right_shift(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    logical_right_shift_0 = builder.logical_right_shift(in0, in1, unit_attrs=unit_attrs)
    # Constrain shift amounts to be within valid range
    shift_tensor_1 = builder._get_golden_tensor(in1)
    dtype_bits = torch.iinfo(shift_tensor_1.shard_at(0).dtype).bits
    # Handle uint32 which doesn't support % operator in PyTorch
    constrained_shift_tensor = shift_tensor_1.apply_shardwise(
        lambda shard: (shard.to(torch.int64) % dtype_bits).to(shard.dtype)
    )

    golden_fn = get_golden_function(ttir.LogicalRightShiftOp)
    output_golden = golden_fn(builder._get_golden_tensor(in0), constrained_shift_tensor)
    builder.set_goldens_from_builder_tensor(
        {in1: constrained_shift_tensor}, {logical_right_shift_0: output_golden}
    )
    return logical_right_shift_0


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
    compile_and_execute_ttir(
        test_fn,
        inputs_shapes=[shape, shape],
        inputs_types=[dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
        randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[equal_indices]

        input_tensor1 = randn_tensor1.to(dtype)
        input_tensor2 = randn_tensor2.to(dtype)

        builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

        return test_fn(in0, in1, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        comparison_ops,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# Unaligned shapes for binary ops
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unaligned_shapes_add(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
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
        return builder.add(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        add,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
    compile_and_execute_ttir(
        test_fn,
        shapes,
        [dtype] * len(shapes),
        test_base=f"{request.node.name}",
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
    compile_and_execute_ttir(
        test_fn,
        shapes,
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_pow(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def hoisted_pow_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return pow(in0, in1, builder, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        hoisted_pow_wrapper,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
