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
    compile_ttir_to_flatbuffer,
    compile_and_execute_ttir,
)

from test_utils import (
    Marks,
    shape_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttir")

enablePrintIR = True


# Matmul runs on the FPU and so needs special care around accuracy checks.
# 1. F32 inputs are truncated into TF32, losing 13 mantissa bits. When positive
#    and negative values with very close abs values are added together, some
#    arithmetic operations will have over 5 orders of magnitude of differences
#    in their operands. TF32 dosn't have this much "dynamic range".
# 2. When the CPU doesn't have native F16/BF16 support, torch will use
#    software-emulated arithmetic operations to generate the matmul golden
#    output, which is too slow.
# Solution: constraint the input range to within (0.001, 0.999) to avoid large
# differences of magnitudes in the calculation.
def create_matmul_constrained_inputs(lhs_shape, rhs_shape):
    def matmul_constrained_inputs(
        in0: Operand, in1: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
    ):
        in_lhs = torch.rand(lhs_shape, dtype=torch.float32) * 0.999 + 0.001
        in_rhs = torch.rand(rhs_shape, dtype=torch.float32) * 0.999 + 0.001
        builder.set_goldens(inputs={in0: in_lhs, in1: in_rhs})
        return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    return matmul_constrained_inputs


def add_f32_tensors(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
):
    """Add two f32 tensors of ones"""
    shape = (256, 256)
    # shape = (256, 256)
    input_0 = torch.rand(shape, dtype=torch.bfloat16)
    input_1 = torch.rand(shape, dtype=torch.bfloat16)
    result = builder.add(in0, in1)
    output_0 = (input_0 + input_1).to(torch.bfloat16)
    builder.set_goldens({in0: input_0, in1: input_1}, {result: output_0})
    return result


@pytest.mark.parametrize(
    "grid", ["override-device-shape=1,1 global-data-format-target=bfp_bf8"]
)
# @pytest.mark.parametrize("dtype", ["global-data-format-target=f32"])
# @pytest.mark.parametrize("grid", ["override-device-shape=1,1"])
@pytest.mark.parametrize("shape", [(256, 256)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_add_f32_tensors(grid: str, shape: Shape, target: str, request):
    """Test add operation with two f32 tensors"""
    options = [grid]

    compile_ttir_to_flatbuffer(
        add_f32_tensors,
        [shape, shape],
        # [torch.float32, torch.float32],  # Two f32 inputs
        [torch.bfloat16, torch.bfloat16],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=enablePrintIR,
    )


@pytest.mark.parametrize(
    "grid", ["override-device-shape=1,1 global-data-format-target=bfp_bf8"]
)
# @pytest.mark.parametrize("grid", ["override-device-shape=1,1"])
@pytest.mark.parametrize("shape", [(256, 256)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_exp_f32(grid: str, shape: Shape, target: str, request, device):
    """Test unary exp operation on f32 tensor"""
    options = [grid]

    def exp_f32(
        in0: Operand,
        builder: TTIRBuilder,
    ):
        shape = (256, 256)
        input_0 = torch.rand(shape, dtype=torch.float32)  # * 0.999 + 0.001
        result = builder.exp(in0)
        output_0 = torch.exp(input_0)
        builder.set_goldens({in0: input_0}, {result: output_0})
        return result

    compile_and_execute_ttir(
        exp_f32,
        [shape],
        [torch.float32],  # Input is f32
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
        # (512, 512, 512),
        # (512, 1024, 1024),
        # (512, 1024, 2048),
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
    )
