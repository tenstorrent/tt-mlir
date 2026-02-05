# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import (
    shapes_list_str,
    shape_str,
)

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("shape", [(128, 128), (4, 128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_matmul(
    shape: Shape,
    dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def matmul(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.matmul(
                in0,
                in1,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                unit_attrs=unit_attrs,
            )

    pipeline_options = []
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (10, 64, 64),
            (64, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("has_bias", [True, False], ids=["with_bias", "without_bias"])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_linear(
    shapes: List[Shape],
    dtype: torch.dtype,
    has_bias: bool,
    transpose_a: bool,
    transpose_b: bool,
    target: str,
    request,
    device,
):
    bias_shape = None
    if has_bias:
        bias_shape = (shapes[1][-1],)

    def module(builder: TTIRBuilder):
        # Set up input shapes and types based on whether bias is used
        if has_bias:
            input_shapes = shapes + [bias_shape]
            input_types = [dtype, dtype, dtype]
        else:
            input_shapes = shapes
            input_types = [dtype, dtype]

        @builder.func(input_shapes, input_types)
        def linear(*args, unit_attrs: Optional[List[str]] = None):
            # The builder is always passed as the last positional argument
            builder = args[-1]
            inputs = args[:-1]

            in0 = inputs[0]
            in1 = inputs[1]
            bias = inputs[2] if len(inputs) > 2 else None

            return builder.linear(
                in0,
                in1,
                bias,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        [(10, 64, 32), (32, 128), (128,)],
        [(10, 20), (20, 30)],
    ],
    ids=["3D_with_bias", "2D_no_bias"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_linear(
    shapes: List[Shape], dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def hoisted_linear(
            *inputs,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder = inputs[-1]
            in0 = inputs[0]
            in1 = inputs[1]
            bias = inputs[2] if len(inputs) > 3 else None
            return builder.linear(in0, in1, bias, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        [(10, 20), (20, 30)],
        [(5, 10, 20), (5, 20, 30)],
    ],
    ids=["standard_2D_matmul", "3D_batched_matmul"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_matmul(
    shapes: List[Shape], dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def hoisted_matmul(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.matmul(in0, in1, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
