# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from golden import get_golden_function
from builder.base.builder_apis import (
    compile_and_execute_ttnn,
)
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
)
from ttmlir.dialects import ttnn


pytestmark = pytest.mark.frontend("ttnn")


def module_add(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def add(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.add(in0, in1)


def module_atan2(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def atan2(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.atan2(in0, in1, unit_attrs=unit_attrs)


def module_divide(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def divide(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.divide(in0, in1, unit_attrs=unit_attrs)


def module_logical_and(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def logical_and(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.logical_and(in0, in1, unit_attrs=unit_attrs)


def module_logical_or(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def logical_or(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.logical_or(in0, in1, unit_attrs=unit_attrs)


def module_logical_xor(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def logical_xor(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.logical_xor(in0, in1, unit_attrs=unit_attrs)


def module_maximum(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def maximum(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.maximum(in0, in1)


def module_minimum(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def minimum(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.minimum(in0, in1, unit_attrs=unit_attrs)


def module_multiply(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def multiply(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.multiply(in0, in1)


def module_remainder(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def remainder(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.remainder(in0, in1, unit_attrs=unit_attrs)


def module_subtract(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def subtract(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.subtract(in0, in1, unit_attrs=unit_attrs)


def module_pow_tensor(builder: TTNNBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def pow_tensor(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.pow_tensor(in0, in1, unit_attrs=unit_attrs)


binary_ops = [
    module_add,
    module_atan2,
    module_divide,
    module_logical_and,
    module_logical_or,
    module_logical_xor,
    module_maximum,
    module_minimum,
    module_multiply,
    module_remainder,
    module_subtract,
    module_pow_tensor,
]


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", binary_ops)
def test_binary_ops(test_fn: Callable, target: str, request, device):
    # FP32 device untilize has precision issues with NaN handling.
    # See: https://github.com/tenstorrent/tt-metal/pull/33904
    if test_fn.__name__ == "module_pow_tensor":
        pytest.xfail(
            "FP32 pow_tensor fails due to tt-metal untilize NaN handling. "
            "See: https://github.com/tenstorrent/tt-metal/pull/33904"
        )
    pipeline_options = []
    compile_and_execute_ttnn(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


binary_bitwise_dtypes = [
    torch.int32,
    torch.uint32,
    torch.uint16,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", binary_bitwise_dtypes, ids=["i32", "u32", "u16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_bitwise_binary_ops_and(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def bitwise_and(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.bitwise_and(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", binary_bitwise_dtypes, ids=["i32", "u32", "u16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_bitwise_binary_ops_or(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def bitwise_or(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.bitwise_or(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", binary_bitwise_dtypes, ids=["i32", "u32", "u16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_bitwise_binary_ops_xor(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def bitwise_xor(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.bitwise_xor(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


binary_logical_shift_dtypes = [
    torch.int32,
    torch.uint32,
    torch.uint16,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", binary_logical_shift_dtypes, ids=["i32", "u32", "u16"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_logical_shift_binary_ops_logical_left_shift(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if dtype == torch.uint16:
        pytest.xfail("uint16 logical left shift op is not supported yet")

    # Binary logical shift ops (int only)
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def logical_left_shift(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Create random shift amounts within valid range using torch tensors
            dtype_bits = torch.iinfo(dtype).bits
            constrained_shift_tensor = torch.randint(
                low=0, high=dtype_bits, size=shape, dtype=dtype
            )
            builder.set_goldens({in1: constrained_shift_tensor})
            return builder.logical_left_shift(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", binary_logical_shift_dtypes, ids=["i32", "u32", "u16"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_logical_shift_binary_ops_logical_right_shift(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def logical_right_shift(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Create random shift amounts within valid range using torch tensors
            dtype_bits = torch.iinfo(dtype).bits
            constrained_shift_tensor = torch.randint(
                low=0, high=dtype_bits, size=shape, dtype=dtype
            )
            builder.set_goldens({in1: constrained_shift_tensor})
            return builder.logical_right_shift(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Binary comparison ops


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_eq(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def eq(
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
            randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[
                equal_indices
            ]

            input_tensor1 = randn_tensor1.to(dtype)
            input_tensor2 = randn_tensor2.to(dtype)

            builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

            return builder.eq(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_ge(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def eq(
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
            randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[
                equal_indices
            ]

            input_tensor1 = randn_tensor1.to(dtype)
            input_tensor2 = randn_tensor2.to(dtype)

            builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

            return builder.ge(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_gt(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def eq(
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
            randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[
                equal_indices
            ]

            input_tensor1 = randn_tensor1.to(dtype)
            input_tensor2 = randn_tensor2.to(dtype)

            builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

            return builder.gt(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_le(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def eq(
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
            randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[
                equal_indices
            ]

            input_tensor1 = randn_tensor1.to(dtype)
            input_tensor2 = randn_tensor2.to(dtype)

            builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

            return builder.le(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_lt(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def eq(
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
            randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[
                equal_indices
            ]

            input_tensor1 = randn_tensor1.to(dtype)
            input_tensor2 = randn_tensor2.to(dtype)

            builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

            return builder.lt(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_ne(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def eq(
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
            randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[
                equal_indices
            ]

            input_tensor1 = randn_tensor1.to(dtype)
            input_tensor2 = randn_tensor2.to(dtype)

            builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

            return builder.ne(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
