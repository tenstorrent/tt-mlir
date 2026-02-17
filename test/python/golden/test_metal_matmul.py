# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from typing import List
from conftest import get_request_kwargs

from test_utils import shape_str

from ttmlir.ir import Context, Location, Module
from ttmlir.passes import ttmetal_to_flatbuffer_bin
from builder.base.builder_runtime import execute_fb
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from golden import GoldenMapTensor

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
def create_matmul_constrained_inputs(lhs_shape, rhs_shape, dtype=torch.float32):
    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [dtype, dtype])
        def matmul_constrained_inputs(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            if dtype == torch.float32:
                in_lhs = torch.rand(lhs_shape, dtype=torch.float32) * 0.999 + 0.001
                in_rhs = torch.rand(rhs_shape, dtype=torch.float32) * 0.999 + 0.001
            else:
                in_lhs = torch.rand(lhs_shape, dtype=torch.bfloat16)
                in_rhs = torch.rand(rhs_shape, dtype=torch.bfloat16)
            builder.set_goldens(inputs={in0: in_lhs, in1: in_rhs})
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    return module


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
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
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
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
    )


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "use_tile_matmul", [True, False], ids=["matmul_tile", "matmul_block"]
)
@pytest.mark.parametrize("enable_l1_acc", [True, False], ids=["l1_acc", "no_l1_acc"])
@pytest.mark.parametrize("target", ["ttmetal"])
# Large matmuls, based on ttnn's matmul benchmarks
def test_matmul_ttnn_shapes_single_buffered(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    use_tile_matmul: bool,
    enable_l1_acc: bool,
    target: str,
    request,
    device,
):
    pcc = 0.99 if dtype == torch.float32 else 0.96
    if (
        dtype == torch.bfloat16
        and not enable_l1_acc
        and shape
        in (
            (2048, 2048, 2048),
            (1024, 2048, 2048),
        )
    ):
        pytest.xfail(reason="bf16 PCC below threshold for these shapes")

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
        f"enable-l1-acc={enable_l1_acc}",
    ]
    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs, rhs, dtype),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        save_artifacts=True,
        skip_exec=getattr(request.node, "skip_exec", False),
        pcc=pcc,
    )


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "use_tile_matmul", [True, False], ids=["matmul_tile", "matmul_block"]
)
@pytest.mark.parametrize("enable_l1_acc", [True, False], ids=["l1_acc", "no_l1_acc"])
@pytest.mark.parametrize("target", ["ttmetal"])
# Large matmuls, based on ttnn's matmul benchmarks
def test_matmul_ttnn_shapes_double_buffered(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    use_tile_matmul: bool,
    enable_l1_acc: bool,
    target: str,
    request,
    device,
):
    pcc = 0.99 if dtype == torch.float32 else 0.96
    if dtype == torch.float32 and shape == (2048, 2048, 2048):
        pytest.xfail(reason="Too large for f32.")

    if (
        dtype == torch.bfloat16
        and not enable_l1_acc
        and shape
        in (
            (2048, 2048, 2048),
            (1024, 2048, 2048),
        )
    ):
        pytest.xfail(reason="bf16 PCC below threshold for these shapes")

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
        f"enable-l1-acc={enable_l1_acc}",
    ]
    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs, rhs, dtype),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        save_artifacts=True,
        skip_exec=getattr(request.node, "skip_exec", False),
        pcc=pcc,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_from_mlir_file(
    target: str,
    request,
    device,
):
    pcc = 0.96

    lhs_shape = (512, 1024)
    rhs_shape = (1024, 1024)

    # Build constrained golden inputs following create_matmul_constrained_inputs.
    in_lhs = torch.rand(lhs_shape, dtype=torch.bfloat16)
    in_rhs = torch.rand(rhs_shape, dtype=torch.bfloat16)

    expected_output = torch.matmul(in_lhs, in_rhs)

    golden_input_output_tensors = {
        0: {
            "input_0": GoldenMapTensor({0: in_lhs}, (1, 1)),
            "input_1": GoldenMapTensor({0: in_rhs}, (1, 1)),
            "output_0": GoldenMapTensor({0: expected_output}, (1, 1)),
        }
    }

    # artifact_dir = os.path.join(
    #     os.path.dirname(__file__),
    #     "..",
    #     "..",
    #     "..",
    #     "builder-artifacts",
    #     "TTIRBuilder",
    #     request.node.name.replace(
    #         "test_matmul_from_ttmetal_mlir", "test_matmul_ttnn_shapes_double_buffered"
    #     ),
    # )
    mlir_path = "/localdev/vtang/tt-mlir/packer-matmul-block-512x1024x1024.mlir"

    if not os.path.exists(mlir_path):
        pytest.skip(f"TTMetal MLIR artifact not found: {mlir_path}")

    with open(mlir_path, "r", encoding="utf-8") as f:
        mlir_text = f.read()

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)

    compiled_bin = ttmetal_to_flatbuffer_bin(module)

    execute_fb(
        compiled_bin,
        golden_input_output_tensors,
        {},
        device=device,
        check_pcc=True,
        pcc=pcc,
    )


@pytest.mark.skip_config(["ttmetal", "p150"], reason="See issue #5341")
@pytest.mark.parametrize(
    "shape",
    [
        (32, 4096, 2048),  # width sharded in0, block sharded in1
        (32768, 32, 32),  # height sharded in0, block sharded in1
        (32, 32, 32768),  # block sharded in0, width sharded in1
        (2048, 4096, 32),  # block sharded in0, height sharded in1
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_1d_shapes(
    shape: tuple[int, ...],
    dtype: torch.dtype,
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

    def module(builder: TTIRBuilder):
        @builder.func([lhs, rhs], [dtype, dtype])
        def matmul_1d(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    options = [
        f"matmul-interchange=2,0,1",
        f"use-tile-matmul=false",
    ]
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        save_artifacts=True,
        skip_exec=getattr(request.node, "skip_exec", False),
    )
