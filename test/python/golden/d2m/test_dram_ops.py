# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Optional

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")

DRAM_PIPELINE_OPTIONS = [
    "default-input-memspace=dram",
    "default-output-memspace=dram",
]

DRAM_FUSION_PIPELINE_OPTIONS = [
    *DRAM_PIPELINE_OPTIONS,
    "enable-elementwise-fusion=true",
]

DRAM_SHAPES = [
    (1, 16),
    (16, 16),
    (32, 32),
    (32, 64),
    (64, 128),
    (128, 128),
    (256, 512),
    (512, 1024),
    (2048, 2048),
]


def dtype_str(dtype: torch.dtype):
    return {
        torch.float32: "f32",
        torch.bfloat16: "bf16",
        torch.int32: "i32",
    }[dtype]


def shape_dtype_param(shape: Shape, dtype: torch.dtype):
    marks = [pytest.mark.skip_config(["sim"])] if dtype == torch.int32 else []
    return pytest.param(
        shape, dtype, marks=marks, id=f"{dtype_str(dtype)}-{shape_str(shape)}"
    )


DRAM_FLOAT_CASES = [
    shape_dtype_param(shape, dtype)
    for shape, dtype in zip(
        DRAM_SHAPES,
        [
            torch.float32,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            torch.float32,
        ],
    )
]

DRAM_NUMERIC_CASES = [
    shape_dtype_param(shape, dtype)
    for shape, dtype in zip(
        DRAM_SHAPES,
        [
            torch.float32,
            torch.bfloat16,
            torch.int32,
            torch.float32,
            torch.bfloat16,
            torch.int32,
            torch.float32,
            torch.bfloat16,
            torch.int32,
        ],
    )
]

DRAM_FUSION_CASES = [
    shape_dtype_param((32, 32), torch.float32),
    shape_dtype_param((128, 128), torch.bfloat16),
]


def get_add_scalar_value(dtype: torch.dtype):
    return 5 if dtype == torch.int32 else 5.0 if dtype == torch.bfloat16 else 2.5


def get_clamp_bounds(dtype: torch.dtype):
    return (0, 3) if dtype == torch.int32 else (-0.5, 0.8)


def compile_dram_test(module, request, device, target: str):
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=list(DRAM_PIPELINE_OPTIONS),
    )


def compile_dram_fusion_test(module, request, device, target: str):
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=list(DRAM_FUSION_PIPELINE_OPTIONS),
    )


@pytest.mark.parametrize("shape,dtype", DRAM_FLOAT_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_unary_relu(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def relu(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[list[str]] = None
        ):
            return builder.relu(in0, unit_attrs=unit_attrs)

    compile_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_FLOAT_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_binary_add(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def add(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[list[str]] = None,
        ):
            return builder.add(in0, in1, unit_attrs=unit_attrs)

    compile_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_FUSION_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_fuse_add_relu_multiply(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape, shape], [dtype, dtype, dtype])
        def add_relu_multiply(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[list[str]] = None,
        ):
            sum_result = builder.add(in0, in1, unit_attrs=unit_attrs)
            relu_result = builder.relu(sum_result, unit_attrs=unit_attrs)
            return builder.multiply(relu_result, in2, unit_attrs=unit_attrs)

    compile_dram_fusion_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_FUSION_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_fuse_converging_branches(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape, shape], [dtype, dtype, dtype])
        def converging_branches(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[list[str]] = None,
        ):
            lhs = builder.relu(in0, unit_attrs=unit_attrs)
            rhs = builder.add(in1, in2, unit_attrs=unit_attrs)
            return builder.multiply(lhs, rhs, unit_attrs=unit_attrs)

    compile_dram_fusion_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_NUMERIC_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_binary_add_scalar(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def add_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[list[str]] = None
        ):
            scalar_value = get_add_scalar_value(dtype)
            scalar = builder.constant(torch.full(shape, scalar_value, dtype=dtype))
            return builder.add(in0, scalar, unit_attrs=unit_attrs)

    compile_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_NUMERIC_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_ternary_clamp_tensor(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape, shape], [dtype, dtype, dtype])
        def clamp_tensor(
            in0: Operand,
            min_tensor: Operand,
            max_tensor: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[list[str]] = None,
        ):
            if dtype == torch.int32:
                input_golden = torch.randint(-5, 10, shape, dtype=dtype)
            else:
                input_golden = torch.rand(shape, dtype=dtype) * 2 - 1
            min_value, max_value = get_clamp_bounds(dtype)
            min_golden = torch.full(shape, min_value, dtype=dtype)
            max_golden = torch.full(shape, max_value, dtype=dtype)
            builder.set_goldens(
                inputs={
                    in0: input_golden,
                    min_tensor: min_golden,
                    max_tensor: max_golden,
                }
            )
            return builder.clamp_tensor(
                in0, min_tensor, max_tensor, unit_attrs=unit_attrs
            )

    compile_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_NUMERIC_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_ternary_clamp_scalar(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def clamp_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[list[str]] = None
        ):
            if dtype == torch.int32:
                input_golden = torch.randint(-5, 10, shape, dtype=dtype)
            else:
                input_golden = torch.rand(shape, dtype=dtype) * 2 - 1
            builder.set_goldens(inputs={in0: input_golden})
            min_value, max_value = get_clamp_bounds(dtype)
            return builder.clamp_scalar(
                in0, min_arg=min_value, max_arg=max_value, unit_attrs=unit_attrs
            )

    compile_dram_test(module, request, device, target)
