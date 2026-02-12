# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Callable, Sequence, Optional

from ttmlir.ir import *
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline
from ttmlir.dialects import ttir

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

from test_metal_matmul import create_matmul_constrained_inputs

from test_utils import (
    shape_str,
)

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("shape", [(512, 512)])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.xfail(
    reason="fp32->bf16 typecast fails due to LLK tiling issue. "
    "See comment at: https://github.com/tenstorrent/tt-metal/issues/35302"
)
def test_triple_exp_f32(shape: Shape, target: str, request, device):
    pipeline_options = ["global-data-format-target=bfp_bf8"]

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def triple_exp_f32(
            in0: Operand,
            builder: TTIRBuilder,
        ):
            input_0 = torch.rand(shape, dtype=torch.float32)
            exp0 = builder.exp(in0)
            tcast0 = builder.typecast(
                exp0, torch.bfloat16, unit_attrs=["preserveDataFormat"]
            )
            exp1 = builder.exp(tcast0)
            tcast1 = builder.typecast(
                exp1, torch.float32, unit_attrs=["preserveDataFormat"]
            )
            exp2 = builder.exp(tcast1)
            output_0 = torch.exp(torch.exp(torch.exp(input_0)))
            builder.set_goldens({in0: input_0}, {exp2: output_0})
            return exp2

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
        save_artifacts=True,
        pcc=0.988,  # Adjusted for bfp8
    )


@pytest.mark.parametrize("shape", [(512, 512)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_exp_f32(shape: Shape, target: str, request, device):
    pipeline_options = ["global-data-format-target=bfp_bf8"]

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def exp_f32(
            in0: Operand,
            builder: TTIRBuilder,
        ):
            input_0 = torch.rand(shape, dtype=torch.float32)
            result = builder.exp(in0)
            output_0 = torch.exp(input_0)
            builder.set_goldens({in0: input_0}, {result: output_0})
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
        save_artifacts=True,
    )


@pytest.mark.parametrize("shape", [(512, 512)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_cos_bf16(shape: Shape, target: str, request, device):
    pipeline_options = ["global-data-format-target=bfp_bf8"]

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.bfloat16])
        def cos_bf16(
            in0: Operand,
            builder: TTIRBuilder,
        ):
            input_0 = torch.rand(shape, dtype=torch.bfloat16)
            result = builder.cos(in0)
            output_0 = torch.cos(input_0).to(torch.bfloat16)
            builder.set_goldens({in0: input_0}, {result: output_0})
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
        save_artifacts=True,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (512, 512, 512),
        (512, 1024, 1024),
        (512, 1024, 2048),
        (1024, 1024, 1024),
        (1024, 1024, 2048),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("use_tile_matmul", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
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
        f"global-data-format-target=bfp_bf8",
    ]
    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs, rhs),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        save_artifacts=True,
        pcc=0.94,  # Adjusted for bfp8
    )
