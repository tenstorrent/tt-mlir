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
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")


@x86_only
@pytest.mark.parametrize(
    "weight_dtype",
    [
        "bfp_bf8",
        "bfp_bf4",
        pytest.param(
            "bfp_bf2",
            marks=pytest.mark.skip_exec(
                ("p150",),
                ("p300",),
                reason="BFP_BFloat2 not supported on Blackhole",
            ),
        ),
    ],
)
@pytest.mark.parametrize("shape", [(4, 128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_matmul_weight_dtype(
    weight_dtype: str,
    shape: Shape,
    dtype: torch.dtype,
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
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    pcc = (
        0.95
        if weight_dtype == "bfp_bf2"
        else 0.98
        if weight_dtype == "bfp_bf4"
        else 0.99
    )
    compile_and_execute_ttir(
        module,
        argument_types_string="matmul=input,parameter",
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[f"experimental-weight-dtype={weight_dtype}"],
        pcc=pcc,
    )


@x86_only
@pytest.mark.parametrize(
    "weight_dtype",
    [
        "bfp_bf8",
        "bfp_bf4",
        pytest.param(
            "bfp_bf2",
            marks=pytest.mark.skip_exec(
                ("p150",),
                ("p300",),
                reason="BFP_BFloat2 not supported on Blackhole",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "shapes",
    [[(10, 64, 64), (64, 64), (64,)]],
    ids=["10x64x64_64x64_bias64"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_linear_weight_dtype(
    weight_dtype: str,
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def linear(*args, unit_attrs: Optional[List[str]] = None):
            builder = args[-1]
            inputs = args[:-1]
            in0 = inputs[0]
            in1 = inputs[1]
            bias = inputs[2] if len(inputs) > 2 else None
            return builder.linear(in0, in1, bias, unit_attrs=unit_attrs)

    pcc = (
        0.95
        if weight_dtype == "bfp_bf2"
        else 0.98
        if weight_dtype == "bfp_bf4"
        else 0.99
    )
    compile_and_execute_ttir(
        module,
        argument_types_string="linear=input,parameter,input",
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[f"experimental-weight-dtype={weight_dtype}"],
        pcc=pcc,
    )
