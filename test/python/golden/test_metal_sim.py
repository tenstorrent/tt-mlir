# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shape_str
from ttir_ops.eltwise.test_ttir_unary import exp, sqrt, cos
from ttir_ops.eltwise.test_ttir_binary import add, multiply, div
from test_metal_matmul import create_matmul_constrained_inputs

pytestmark = pytest.mark.frontend("ttir")
SIM_TARGET = "ttmetal"
SMALL_SHAPE = (128, 128)


@pytest.mark.parametrize("shape", [SMALL_SHAPE], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", [SIM_TARGET])
@pytest.mark.parametrize("test_fn", [exp, sqrt, cos], ids=["exp", "sqrt", "cos"])
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def unary_op(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return test_fn(in0, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[],
    )


@pytest.mark.parametrize("shape", [SMALL_SHAPE], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", [SIM_TARGET])
@pytest.mark.parametrize(
    "test_fn", [add, multiply, div], ids=["add", "multiply", "div"]
)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            return test_fn(in0, in1, builder)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[],
    )


@pytest.mark.parametrize(
    "shape",
    [
        (512, 512, 512),
        (1024, 1024, 2048),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("use_tile_matmul", [True, False])
@pytest.mark.parametrize("target", [SIM_TARGET])
def test_matmul_f32(
    shape: tuple[int, ...],
    use_tile_matmul: bool,
    target: str,
    request,
    device,
):
    lhs = (
        shape[0],
        shape[1],
    )
    rhs = (
        shape[1],
        shape[2],
    )

    options = [
        f"matmul-interchange=2,0,1",
        f"use-tile-matmul={use_tile_matmul}",
    ]
    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs, rhs, torch.float32),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        save_artifacts=True,
        pcc=0.99,
    )
