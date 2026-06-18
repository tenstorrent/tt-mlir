# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import pytest
import torch
from typing import List, Optional

from builder.base.builder_utils import Operand, Shape
from builder.base.builder_apis import compile_and_execute_ttir
from d2m.test_matmul import create_matmul_constrained_inputs
from d2m.test_reductions import (
    _reduction_atol,
    create_reductions_constrained_inputs,
)
from builder.ttir.ttir_builder import TTIRBuilder
from conftest import get_request_kwargs
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
    (1, 15),
    (17, 1),
    (73, 467),
    (449, 89),
    (512, 8192),
    (16384, 256),
    (2048, 2048),
    (23, 383, 457),
    (5, 3, 439, 373),
]

DRAM_1D_SHAPES = [
    (32,),
    (73,),
    (128,),
]


def dtype_str(dtype: torch.dtype):
    return {
        torch.float32: "f32",
        torch.bfloat16: "bf16",
        torch.int32: "i32",
    }[dtype]


def shape_dtype_param(shape: Shape, dtype: torch.dtype):
    marks = [pytest.mark.skip_config(["n150", "sim"])] if dtype == torch.int32 else []
    return pytest.param(
        shape, dtype, marks=marks, id=f"{dtype_str(dtype)}-{shape_str(shape)}"
    )


DRAM_FLOAT_CASES = [
    shape_dtype_param(shape, dtype)
    for shape, dtype in itertools.product(DRAM_SHAPES, [torch.float32, torch.bfloat16])
]

DRAM_1D_FLOAT_CASES = [
    shape_dtype_param(shape, dtype)
    for shape, dtype in itertools.product(
        DRAM_1D_SHAPES, [torch.float32, torch.bfloat16]
    )
]

DRAM_NUMERIC_CASES = [
    shape_dtype_param(shape, dtype)
    for shape, dtype in itertools.product(
        DRAM_SHAPES, [torch.float32, torch.bfloat16, torch.int32]
    )
]

DRAM_1D_NUMERIC_CASES = [
    shape_dtype_param(shape, dtype)
    for shape, dtype in itertools.product(
        DRAM_1D_SHAPES, [torch.float32, torch.bfloat16, torch.int32]
    )
]

DRAM_FUSION_CASES = [
    shape_dtype_param((32, 32), torch.float32),
    shape_dtype_param((128, 128), torch.bfloat16),
]

DRAM_BROADCAST_CASES = [
    pytest.param(
        (1, 128, 1),
        [1, 1, 2560],
        torch.float32,
        id="f32-1x128x1-to-1x128x2560",
    ),
    pytest.param(
        (1, 1, 2560),
        [1, 128, 1],
        torch.bfloat16,
        id="bf16-1x1x2560-to-1x128x2560",
    ),
    pytest.param(
        (128, 1, 64),
        [1, 32, 1],
        torch.bfloat16,
        id="bf16-128x1x64-to-128x32x64",
    ),
    pytest.param(
        (128, 1, 64),
        [1, 8, 1],
        torch.bfloat16,
        id="bf16-128x1x64-to-128x8x64",
    ),
    pytest.param(
        (1, 128, 32, 1),
        [1, 1, 1, 128],
        torch.float32,
        id="f32-1x128x32x1-to-1x128x32x128",
    ),
    pytest.param(
        (1, 1, 1, 128),
        [1, 128, 32, 1],
        torch.bfloat16,
        id="bf16-1x1x1x128-to-1x128x32x128",
    ),
    pytest.param(
        (1, 64),
        [128, 1],
        torch.float32,
        id="f32-1x64-to-128x64",
    ),
    pytest.param(
        (1, 128, 8, 1),
        [1, 1, 1, 128],
        torch.float32,
        id="f32-1x128x8x1-to-1x128x8x128",
    ),
    pytest.param(
        (1, 1, 1, 128),
        [1, 128, 8, 1],
        torch.bfloat16,
        id="bf16-1x1x1x128-to-1x128x8x128",
    ),
    pytest.param(
        (128, 1),
        [1, 64],
        torch.float32,
        id="f32-128x1-to-128x64",
    ),
]


def get_add_scalar_value(dtype: torch.dtype):
    return 5 if dtype == torch.int32 else 5.0 if dtype == torch.bfloat16 else 2.5


def get_clamp_bounds(dtype: torch.dtype):
    return (0, 3) if dtype == torch.int32 else (-0.5, 0.8)


def compile_and_execute_dram_test(
    module, request, device, target: str, **compile_kwargs
):
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=list(DRAM_PIPELINE_OPTIONS),
        **compile_kwargs,
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
def test_dram_unary_neg(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def neg(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.neg(in0, unit_attrs=unit_attrs)

    compile_and_execute_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_1D_FLOAT_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_1d_unary_neg(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def neg(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.neg(in0, unit_attrs=unit_attrs)

    compile_and_execute_dram_test(module, request, device, target)


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
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.add(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_1D_NUMERIC_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_1d_binary_add(
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
            return builder.add(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_dram_test(module, request, device, target)


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
            unit_attrs: Optional[List[str]] = None,
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
            unit_attrs: Optional[List[str]] = None,
        ):
            lhs = builder.relu(in0, unit_attrs=unit_attrs)
            rhs = builder.add(in1, in2, unit_attrs=unit_attrs)
            return builder.multiply(lhs, rhs, unit_attrs=unit_attrs)

    compile_dram_fusion_test(module, request, device, target)


@pytest.mark.parametrize("shape,broadcast_dimensions,dtype", DRAM_BROADCAST_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_broadcast(
    shape: Shape,
    broadcast_dimensions: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def broadcast(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.broadcast(
                in0,
                broadcast_dimensions=broadcast_dimensions,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_NUMERIC_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_binary_add_scalar(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def add_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            scalar_value = get_add_scalar_value(dtype)
            scalar = builder.full(shape, dtype, scalar_value, unit_attrs=unit_attrs)
            return builder.add(in0, scalar, unit_attrs=unit_attrs)

    compile_and_execute_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_1D_NUMERIC_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_1d_binary_add_scalar(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def add_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            scalar_value = get_add_scalar_value(dtype)
            scalar = builder.full(shape, dtype, scalar_value, unit_attrs=unit_attrs)
            return builder.add(in0, scalar, unit_attrs=unit_attrs)

    compile_and_execute_dram_test(module, request, device, target)


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
            unit_attrs: Optional[List[str]] = None,
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

    compile_and_execute_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape,dtype", DRAM_NUMERIC_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_ternary_clamp_scalar(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def clamp_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
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

    compile_and_execute_dram_test(module, request, device, target)


DRAM_MATMUL_CASES = [
    pytest.param(
        (32, 128),
        (128, 64),
        torch.float32,
        id="f32-32x128x64",
    ),
    pytest.param(
        (128, 320),
        (320, 768),
        torch.bfloat16,
        id="bf16-128x320x768",
    ),
    pytest.param(
        (2, 64, 160),
        (2, 160, 96),
        torch.float32,
        id="f32-batch2-64x160x96",
    ),
]

DRAM_REDUCTION_CASES = [
    pytest.param(
        (1, 128, 320),
        "sum",
        [2],
        False,
        torch.float32,
        id="f32-sum-1x128x320-dim2",
    ),
    pytest.param(
        (2, 128, 160),
        "mean",
        [2],
        True,
        torch.float32,
        id="f32-mean-2x128x160-dim2-keep",
    ),
    pytest.param(
        (4, 64, 128),
        "sum",
        [1],
        True,
        torch.bfloat16,
        id="bf16-sum-4x64x128-dim1-keep",
    ),
]


@pytest.mark.parametrize("lhs_shape,rhs_shape,dtype", DRAM_MATMUL_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_matmul(
    lhs_shape: Shape,
    rhs_shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    compile_and_execute_dram_test(
        create_matmul_constrained_inputs(lhs_shape, rhs_shape, dtype),
        request,
        device,
        target,
        pcc=0.99,
    )


@pytest.mark.parametrize(
    "shape,reduce_type,dim_arg,keep_dim,dtype", DRAM_REDUCTION_CASES
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_reduction(
    shape: Shape,
    reduce_type: str,
    dim_arg: List[int],
    keep_dim: bool,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    compile_and_execute_dram_test(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim, dtype
        ),
        request,
        device,
        target,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
        pcc=0.99,
    )


DRAM_EMBEDDING_CASES = [
    pytest.param(
        (1, 32),
        (1024, 32),
        torch.uint32,
        torch.bfloat16,
        id="ui32_indices_bf16_table_1x32_1024x32",
    ),
    pytest.param(
        (1, 32),
        (1024, 40),
        torch.uint32,
        torch.bfloat16,
        id="ui32_indices_bf16_table_1x32_1024x40",
    ),
    pytest.param(
        (1, 32),
        (1024, 32),
        torch.uint32,
        torch.int32,
        marks=pytest.mark.skip_config(["n150", "sim"]),
        id="ui32_indices_i32_table_1x32_1024x32",
    ),
    pytest.param(
        (1, 32),
        (8192, 64),
        torch.uint32,
        torch.float32,
        id="ui32_indices_f32_table_1x32_8192x64",
    ),
    pytest.param(
        (1, 128),
        (40960, 128),
        torch.uint32,
        torch.bfloat16,
        id="ui32_indices_bf16_table_1x128_40960x128",
    ),
]


@pytest.mark.parametrize(
    "indices_shape,weight_shape,indices_dtype,weight_dtype", DRAM_EMBEDDING_CASES
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_embedding(
    indices_shape: Shape,
    weight_shape: Shape,
    indices_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    target: str,
    request,
    device,
):
    num_indices = math.prod(indices_shape)

    def module(builder: TTIRBuilder):
        @builder.func([indices_shape, weight_shape], [indices_dtype, weight_dtype])
        def embedding(
            indices: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            valid_indices = (
                torch.arange(num_indices, dtype=torch.int64).reshape(indices_shape) * 7
            ) % weight_shape[0]
            builder.set_goldens(inputs={indices: valid_indices.to(indices_dtype)})
            return builder.embedding(indices, weight, unit_attrs=unit_attrs)

    compile_and_execute_dram_test(module, request, device, target)
