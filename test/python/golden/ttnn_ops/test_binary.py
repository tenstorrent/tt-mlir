# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only
from builder.base.builder import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base import get_golden_function
from builder.base.builder_utils import (
    compile_and_execute_ttnn,
)
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
)
from ttmlir.dialects import ttnn


pytestmark = pytest.mark.frontend("ttnn")


# Binary ops
def add(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.add(in0, in1, unit_attrs=unit_attrs)


def atan2(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.atan2(in0, in1, unit_attrs=unit_attrs)


def divide(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.divide(in0, in1, unit_attrs=unit_attrs)


def logical_and(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_and(in0, in1, unit_attrs=unit_attrs)


def logical_or(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_or(in0, in1, unit_attrs=unit_attrs)


def logical_xor(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.logical_xor(in0, in1, unit_attrs=unit_attrs)


def maximum(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.maximum(in0, in1, unit_attrs=unit_attrs)


def minimum(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.minimum(in0, in1, unit_attrs=unit_attrs)


def multiply(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


def remainder(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.remainder(in0, in1, unit_attrs=unit_attrs)


def subtract(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.subtract(in0, in1, unit_attrs=unit_attrs)


binary_ops = [
    add,
    atan2,
    divide,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    remainder,
    subtract,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize("test_fn", binary_ops)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    pipeline_options = []
    compile_and_execute_ttnn(
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
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_and(in0, in1, unit_attrs=unit_attrs)


def bitwise_or(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.bitwise_or(in0, in1, unit_attrs=unit_attrs)


def bitwise_xor(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
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
    compile_and_execute_ttnn(
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
    builder: TTNNBuilder,
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

    golden_fn = get_golden_function(ttnn.LogicalLeftShiftOp)
    output_golden = golden_fn(builder._get_golden_tensor(in0), constrained_shift_tensor)
    builder.set_goldens_from_builder_tensor(
        {in1: constrained_shift_tensor}, {logical_left_shift_0: output_golden}
    )
    return logical_left_shift_0


def logical_right_shift(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
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

    golden_fn = get_golden_function(ttnn.LogicalRightShiftOp)
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
    compile_and_execute_ttnn(
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
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.eq(in0, in1, unit_attrs=unit_attrs)


def ge(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.ge(in0, in1, unit_attrs=unit_attrs)


def gt(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.gt(in0, in1, unit_attrs=unit_attrs)


def le(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.le(in0, in1, unit_attrs=unit_attrs)


def lt(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.lt(in0, in1, unit_attrs=unit_attrs)


def ne(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
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
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize("test_fn", binary_comparison_ops)
def test_comparison_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def comparison_ops(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
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

    compile_and_execute_ttnn(
        comparison_ops,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
