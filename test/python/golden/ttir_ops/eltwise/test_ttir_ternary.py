# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
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


# Ternary ops
def module_where(dtype: torch.dtype):
    def _module_where(builder: TTIRBuilder):
        @builder.func([(128, 128), (128, 128), (128, 128)], [dtype] * 3)
        def where(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # in0 is the condition tensor, should be filled with 0s or 1s
            condition_tensor = torch.randint(0, 2, (128, 128), dtype=dtype)
            builder.set_goldens(inputs={in0: condition_tensor})
            return builder.where(in0, in1, in2, unit_attrs=unit_attrs)

    return _module_where


def module_clamp_tensor(dtype: torch.dtype):
    def _module_clamp_tensor(builder: TTIRBuilder):
        @builder.func([(128, 128), (128, 128), (128, 128)], [dtype] * 3)
        def clamp_tensor(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.clamp_tensor(in0, in1, in2, unit_attrs=unit_attrs)

    return _module_clamp_tensor


ternary_ops = [
    module_where,
    module_clamp_tensor,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
@pytest.mark.parametrize("test_fn", ternary_ops)
def test_ternary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    pipeline_options = []
    compile_and_execute_ttir(
        test_fn(dtype),
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


# Ternary eltwise ops with implicit broadcasting
@pytest.mark.parametrize(
    "shapes",
    [
        # 2D shapes
        [(128, 128), (1, 128), (128, 128)],
        [(32, 64), (32, 64), (1, 64)],
        [(1, 32), (64, 32), (64, 1)],
        # 3D shapes
        [(1, 16, 32), (8, 16, 32), (8, 16, 32)],
        [(8, 16, 32), (1, 16, 32), (8, 16, 32)],
        [(8, 16, 32), (8, 16, 32), (1, 16, 32)],
        [(8, 16, 32), (1, 1, 32), (1, 1, 32)],
        [(1, 1, 32), (8, 16, 32), (1, 1, 32)],
        [(1, 1, 32), (1, 1, 32), (8, 16, 32)],
        [(1, 16, 32), (8, 1, 32), (8, 16, 1)],
        [(1, 4, 1), (1, 4, 768), (1, 1, 1)],
        # 4D shapes
        [(1, 1, 1, 4), (1, 1, 1, 1), (1, 1, 1, 1)],
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize(
    "input_dtypes",
    [
        pytest.param((torch.float32, torch.float32, torch.float32), id="f32-f32-f32"),
        pytest.param((torch.float32, torch.int32, torch.int32), id="f32-i32-i32"),
        pytest.param(
            (torch.bfloat16, torch.bfloat16, torch.bfloat16), id="bf16-bf16-bf16"
        ),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_ternary_eltwise_ops_implicit_broadcast(
    shapes: List[Shape],
    input_dtypes: Tuple[torch.dtype, torch.dtype, torch.dtype],
    target: str,
    request,
    device,
):
    # Check if any shape is not 2D
    is_non_2d = any(len(shape) != 2 for shape in shapes)
    if target == "ttmetal" and is_non_2d:
        pytest.xfail(
            "non-2D shapes bcast not yet supported on ttmetal backend, issue here: https://github.com/tenstorrent/tt-mlir/issues/3023"
        )
    if target == "ttmetal" and input_dtypes[0] != input_dtypes[1]:
        pytest.xfail(
            "different input dtypes not yet supported on ttmetal backend, issue here: https://github.com/tenstorrent/tt-mlir/issues/6289"
        )

    dtype1, dtype2, dtype3 = input_dtypes

    def module_implicit_broadcast(builder: TTIRBuilder):
        @builder.func(shapes, [dtype1, dtype2, dtype3])
        def where(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            condition_tensor = torch.randint(0, 2, shapes[0], dtype=dtype1)
            builder.set_goldens(inputs={in0: condition_tensor})
            return builder.where(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module_implicit_broadcast,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("max_arg,min_arg", [(0.8, -0.5)])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_clamp_scalar(shape: Shape, max_arg, min_arg, target: str, request, device):
    def module_clamp_scalar(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def clamp_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            input_tensor = torch.rand(shape, dtype=torch.float32) * 2 - 1
            builder.set_goldens(inputs={in0: input_tensor})
            return builder.clamp_scalar(
                in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module_clamp_scalar,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "max_arg,min_arg",
    [(3, 0)],
    ids=["i32"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_clamp_scalar_i32(shape: Shape, max_arg, min_arg, target: str, request, device):
    def module_clamp_scalar(builder: TTIRBuilder):
        @builder.func([shape], [torch.int32])
        def clamp_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            input_tensor = torch.randint(-5, 10, shape, dtype=torch.int32)
            builder.set_goldens(inputs={in0: input_tensor})
            return builder.clamp_scalar(
                in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module_clamp_scalar,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("max_arg,min_arg", [(0.8, -0.5)])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_clamp_scalar(
    shape: Shape, max_arg: float, min_arg: float, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_clamp_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            # Set input values explicitly in range [-1, 1]
            input_tensor = torch.rand(shape, dtype=torch.float32) * 2 - 1
            builder.set_goldens(inputs={in0: input_tensor})
            return builder.clamp_scalar(
                in0,
                max_arg=max_arg,
                min_arg=min_arg,
                unit_attrs=["ttir.should_hoist"],
            )

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
        @builder.func([shape, shape, shape], [dtype, dtype, dtype])
        def ternary_1d(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # in0 is the condition tensor, should be filled with 0s or 1s
            condition_tensor = torch.randint(0, 2, shape, dtype=dtype)
            builder.set_goldens(inputs={in0: condition_tensor})
            return builder.where(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
