# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from test_utils import shape_str

from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


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


@pytest.mark.parametrize("m", [2])
@pytest.mark.parametrize("k", [4])
@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("target", ["ttmetal"])
# Single core matmuls, 8 output tiles per core max
def test_matmul_single_core_8otpc(m: int, k: int, n: int, target: str, request, device):
    tile_size = 32
    lhs = (
        m * tile_size,
        k * tile_size,
    )
    rhs = (
        k * tile_size,
        n * tile_size,
    )

    options = [
        f"override-device-shape=1,1",
        f"num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs, rhs),
        [lhs, rhs],
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("m", [3, 6, 9])
@pytest.mark.parametrize("k", [4])
@pytest.mark.parametrize("n", [3, 6])
@pytest.mark.parametrize("target", ["ttmetal"])
# Multi core matmuls, 8 output tiles per core max
def test_matmul_multi_core_8otpc(m: int, k: int, n: int, target: str, request, device):
    tile_size = 32
    lhs = (
        m * tile_size,
        k * tile_size,
    )
    rhs = (
        k * tile_size,
        n * tile_size,
    )

    options = [
        f"num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs, rhs),
        [lhs, rhs],
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Temporary testcase to verify larger testcases compile on p150 (execution hangs, but we want to cover non-square grid cases)
@pytest.mark.only_config(["ttmetal", "p150"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_p150_grid_selection_compile_only(target: str, request, device):
    from builder.base.builder_utils import compile_ttir_to_flatbuffer

    tile_size = 32
    lhs = (1024, 1024)
    rhs = (1024, 1024)

    options = [
        f"num-stream-buffers=1",
    ]

    # Compile only - verify IR is well-formed with correct dim_alignments, etc.
    mlir_path = compile_ttir_to_flatbuffer(
        create_matmul_constrained_inputs(lhs, rhs),
        [lhs, rhs],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        system_desc_path=request.config.getoption("--sys-desc"),
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
        (1024, 2048, 2048),
        (2048, 2048, 2048),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("use_tile_matmul", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
# Large matmuls, based on ttnn's matmul benchmarks
def test_matmul_ttnn_shapes_single_buffered(
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
        f"num-stream-buffers=1",
        f"use-tile-matmul={use_tile_matmul}",
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
