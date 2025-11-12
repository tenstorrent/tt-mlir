# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Callable, Sequence, Optional

from ttmlir.ir import *
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline
from ttmlir.dialects import ttir

from builder.base.builder import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import (
    compile_and_execute_ttir,
)

from test_metal_matmul import create_matmul_constrained_inputs

from test_utils import (
    shape_str,
)

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("grid", ["global-data-format-target=bfp_bf8"])
@pytest.mark.parametrize("shape", [(512, 512)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_exp_f32(grid: str, shape: Shape, target: str, request, device):
    """Test unary exp operation on f32 tensor"""
    options = [grid]

    def exp_f32(
        in0: Operand,
        builder: TTIRBuilder,
    ):
        shape = (512, 512)
        input_0 = torch.rand(shape, dtype=torch.float32)
        result = builder.exp(in0)
        output_0 = torch.exp(input_0)
        builder.set_goldens({in0: input_0}, {result: output_0})
        return result

    compile_and_execute_ttir(
        exp_f32,
        [shape],
        [torch.float32],
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=False,
    )


@pytest.mark.parametrize("grid", ["global-data-format-target=bfp_bf8"])
@pytest.mark.parametrize("shape", [(512, 512)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_cos_bf16(grid: str, shape: Shape, target: str, request, device):
    """Test unary cos operation on bf16 tensor"""
    options = [grid]

    def cos_bf16(
        in0: Operand,
        builder: TTIRBuilder,
    ):
        shape = (512, 512)
        input_0 = torch.rand(shape, dtype=torch.bfloat16)
        result = builder.cos(in0)
        output_0 = torch.cos(input_0).to(torch.bfloat16)
        builder.set_goldens({in0: input_0}, {result: output_0})
        return result

    compile_and_execute_ttir(
        cos_bf16,
        [shape],
        [torch.bfloat16],
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=False,
    )


@pytest.mark.skip_config(["ttmetal", "p150"], reason="See issue #5341")
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
# Large matmuls, based on ttnn's matmul benchmarks
def test_matmul_ttnn_shapes_double_buffered(
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
        [lhs, rhs],
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        pcc=0.94,  # Adjusted for bfp8
    )
