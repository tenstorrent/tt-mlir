# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List
from conftest import get_request_kwargs

from test_utils import SkipIf, shape_str

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


# Matmul runs on the FPU and so needs special care around accuracy checks.
# 1. F32 inputs are truncated into TF32, losing 13 mantissa bits. When positive
#    and negative values with very close abs values are added together, some
#    arithmetic operations will have over 5 orders of magnitude of differences
#    in their operands. TF32 doesn't have this much "dynamic range".
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


def module_batch_matmul(builder: TTIRBuilder):
    @builder.func([(2, 32, 64), (2, 64, 32)], [torch.float32, torch.float32])
    def batch_matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return builder.matmul(in0, in1)


@pytest.mark.parametrize(
    "collapse_tensors", [True, False], ids=["collapsed", "non_collapsed"]
)
@pytest.mark.parametrize("target", ["ttmetal"], ids=["ttmetal"])
def test_matmul_collapse_tensors(
    collapse_tensors: bool,
    target: str,
    request,
    device,
):
    pipeline_options = f"{{collapse-tensors-2d={str(collapse_tensors).lower()}}}"
    pipeline = f"ttir-to-ttmetal-pipeline{pipeline_options}"

    compile_and_execute_ttir(
        module_batch_matmul,
        target=target,
        custom_pipeline=pipeline,
        test_base=f"{request.node.name}_{'collapsed' if collapse_tensors else 'non_collapsed'}",
        device=device,
    )


def get_allocator_policy_override(
    shape: tuple[int, ...], dtype: torch.dtype, enable_l1_acc: bool
) -> list[str]:
    if (
        dtype == torch.bfloat16
        and enable_l1_acc
        and shape in ((1024, 2048, 2048), (2048, 2048, 2048))
    ):
        # `auto` over-splits the reduction panel for these large bf16 matmuls,
        # which increases partial accumulations and hurts PCC.
        # TODO (anuragsingh): Revert this to the default allocator policy once precision issues are fixed.
        # Issue here: https://github.com/tenstorrent/tt-mlir/issues/7656
        return ["test-buffer-size-policy=max"]
    return []


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
    options.extend(get_allocator_policy_override(shape, dtype, enable_l1_acc))
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
    options.extend(get_allocator_policy_override(shape, dtype, enable_l1_acc))
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
        (32, 4096, 2048),  # width sharded in0, block sharded in1
        (32768, 32, 32),  # height sharded in0, block sharded in1
        (32, 32, 32768),  # block sharded in0, width sharded in1
        (2048, 4096, 32),  # block sharded in0, height sharded in1
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_1d_shapes(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    if dtype == torch.float32 and shape in ((32768, 32, 32), (32, 32, 32768)):
        pytest.xfail(reason="Too large for f32.")

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


# ---------------------------------------------------------------------------
# Higher-rank matmul tests (3D / 4D batched matmul). These validate that the
# fix for issue #6648 correctly handles batch dimensions without collapsing
# them to 2D.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        # 3D batched matmul: [B, M, K] x [B, K, N] -> [B, M, N]
        ((2, 128, 96), (2, 96, 64)),
        ((4, 64, 128), (4, 128, 64)),
        ((8, 128, 96), (8, 96, 64)),
    ],
    ids=["batch2_128x96x64", "batch4_64x128x64", "batch8_128x96x64"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_3d_batched(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    target: str,
    request,
    device,
):
    options = [
        "num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs_shape, rhs_shape),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        # 4D batched matmul: [B0, B1, M, K] x [B0, B1, K, N] -> [B0, B1, M, N]
        ((32, 8, 32, 128), (32, 8, 128, 128)),
        ((2, 4, 64, 96), (2, 4, 96, 64)),
        ((4, 2, 128, 64), (4, 2, 64, 128)),
    ],
    ids=["batch32x8_32x128x128", "batch2x4_64x96x64", "batch4x2_128x64x128"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_4d_batched(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    target: str,
    request,
    device,
):
    options = [
        "num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs_shape, rhs_shape),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        # Small tile sizes for single core
        ((2, 32, 64), (2, 64, 32)),
        ((4, 64, 64), (4, 64, 64)),
    ],
    ids=["batch2_small", "batch4_small"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_3d_single_core(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    target: str,
    request,
    device,
):
    options = [
        "override-device-shape=1,1",
        "num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs_shape, rhs_shape),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        ((4, 256, 256), (4, 256, 256)),
        ((8, 128, 256), (8, 256, 128)),
    ],
    ids=["batch4_256x256x256", "batch8_128x256x128"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_3d_multi_core(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    target: str,
    request,
    device,
):
    options = [
        "num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs_shape, rhs_shape),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
@pytest.mark.parametrize("target", ["ttmetal" | SkipIf("sim")])
def test_bfp8_matmul_f32(
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
        "matmul-interchange=2,0,1",
        f"use-tile-matmul={use_tile_matmul}",
        "global-data-format-target=bfp_bf8",
    ]
    if shape in (
        (512, 1024, 1024),
        (512, 1024, 2048),
        (1024, 1024, 1024),
        (1024, 1024, 2048),
    ):
        # `auto` shrinks the local reduction panel for these larger matmuls.
        # TODO (anuragsingh): Revert this to the default allocator policy once precision issues are fixed.
        # Issue here: https://github.com/tenstorrent/tt-mlir/issues/7656
        options.append("test-buffer-size-policy=max")
    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs, rhs),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        save_artifacts=True,
        pcc=0.94,
    )


@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("k", [4, 8])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_allocate_matmul(m: int, k: int, n: int, target: str, request, device):
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
        "override-device-shape=1,1",
        "num-stream-buffers=1",
        # Request the allocator to attempt to minimize stream buffer sizes
        # and reblock streams accordingly.
        "test-buffer-size-policy=min",
    ]

    compile_and_execute_ttir(
        create_matmul_constrained_inputs(lhs, rhs),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
    )
