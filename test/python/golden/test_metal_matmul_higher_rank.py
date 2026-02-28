# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from test_utils import shape_str

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


# Similar to test_metal_matmul.py, we constrain inputs to avoid numerical issues
# with TF32 truncation and dynamic range limitations in matmul operations.
def create_higher_rank_matmul_constrained_inputs(lhs_shape, rhs_shape):
    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch.float32, torch.float32])
        def matmul_higher_rank(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            in_lhs = torch.rand(lhs_shape, dtype=torch.float32) * 0.999 + 0.001
            in_rhs = torch.rand(rhs_shape, dtype=torch.float32) * 0.999 + 0.001
            builder.set_goldens(inputs={in0: in_lhs, in1: in_rhs})
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    return module


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
    """Test 3D batched matmul operations.

    This validates that the fix for issue #6648 correctly handles
    batch dimensions without collapsing them to 2D.
    """
    options = [
        "num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_higher_rank_matmul_constrained_inputs(lhs_shape, rhs_shape),
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
        # This is the original issue #6648 case
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
    """Test 4D batched matmul operations.

    This validates that the fix for issue #6648 correctly handles
    multiple batch dimensions without collapsing them to 2D.
    """
    options = [
        "num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_higher_rank_matmul_constrained_inputs(lhs_shape, rhs_shape),
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
    """Test 3D batched matmul on single core.

    Small matmuls that fit on a single core to verify the basic
    correctness of the higher-rank matmul implementation.
    """
    options = [
        "override-device-shape=1,1",
        "num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_higher_rank_matmul_constrained_inputs(lhs_shape, rhs_shape),
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
        # Larger matmuls for multi-core testing
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
    """Test 3D batched matmul on multiple cores.

    Larger matmuls that require multi-core distribution to verify
    the higher-rank matmul works correctly with grid-based execution.
    """
    options = [
        "num-stream-buffers=1",
    ]

    compile_and_execute_ttir(
        create_higher_rank_matmul_constrained_inputs(lhs_shape, rhs_shape),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
