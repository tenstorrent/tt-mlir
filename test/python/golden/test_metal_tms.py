# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TM (Tensor Manipulation) operations on TTMetal backend.

This test suite validates TM operations like permute, transpose, reshape, etc.
"""

import pytest
import torch
from typing import List, Tuple

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")

# Skip reason for shapes with slow compilation due to inefficient coalescing factor calculation
SLOW_COMPILE_SKIP = pytest.mark.skip(
    reason="Slow compilation due to inefficient calculateCoalescingFactors, see https://github.com/tenstorrent/tt-mlir/issues/6375"
)

# Skip for 1D tensor shapes that are not yet supported
ONE_D_SKIP = pytest.mark.skip(
    reason="1D tensor reshapes not yet supported, see https://github.com/tenstorrent/tt-mlir/issues/6376"
)

# Skip for NOC read issue
NOC_ISSUE_SKIP = pytest.mark.skip(
    reason="NOC read issue, see https://github.com/tenstorrent/tt-mlir/issues/6377"
)

# ==================== PERMUTE TESTS ====================


@pytest.mark.parametrize(
    "shape, permutation",
    [
        # 3d inner permutes
        [(3, 32, 32), [0, 2, 1]],
        [(3, 32, 64), [0, 2, 1]],
        [(1, 32, 64), [0, 2, 1]],
        # 4d inner permutes
        [(5, 7, 2, 32), [0, 1, 3, 2]],
        [(5, 7, 2, 64), [0, 1, 3, 2]],
        [(5, 7, 2, 128), [0, 1, 3, 2]],
        # 3d inner permutes (llama-like)
        [(1, 50, 12), [0, 2, 1]],
        [(32, 12, 100), [0, 2, 1]],
        [(32, 11, 64), [0, 2, 1]],
        # 4d inner permutes (llama-like)
        [(1, 32, 12, 100), [0, 1, 3, 2]],
        [(1, 32, 11, 64), [0, 1, 3, 2]],
        [(1, 8, 11, 64), [0, 1, 3, 2]],
        # 3d outer permutes
        [(3, 32, 32), [1, 0, 2]],
        [(3, 32, 64), [1, 0, 2]],
        [(1, 32, 64), [1, 0, 2]],
        # 3d outer permutes (llama-like)
        [(18, 24, 128), [1, 0, 2]],
        [(18, 8, 128), [1, 0, 2]],
        [(128, 24, 128), [1, 0, 2]],
        [(128, 8, 128), [1, 0, 2]],
        # 4d outer permutes
        [(1, 32, 31, 32), [0, 2, 1, 3]],
        [(1, 32, 1, 32), [0, 2, 1, 3]],
        [(5, 7, 2, 32), [0, 2, 1, 3]],
        # 4d outer permutes (llama-like)
        [(1, 18, 24, 128), [0, 2, 1, 3]],
        [(1, 18, 8, 128), [0, 2, 1, 3]],
        [(1, 128, 24, 128), [0, 2, 1, 3]],
        [(1, 128, 8, 128), [0, 2, 1, 3]],
        # 5d outer permutes
        [(1, 3, 3, 3, 3), [0, 2, 1, 3, 4]],
        [(1, 3, 3, 3, 3), [0, 2, 1, 3, 4]],
        [(5, 7, 2, 3, 3), [0, 2, 1, 3, 4]],
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_permute(shape: Shape, permutation: List[int], target: str, request, device):
    """Test permute operations with abs on TTMetal backend."""

    def permute_module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def permute(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            return builder.permute(in0, permutation=permutation)

    compile_and_execute_ttir(
        permute_module,
        target=target,
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '}}}",
    )


# ==================== RESHAPE TESTS ====================

# Test shapes: (input_shape, output_shape)
# All shapes must have the same total number of elements.
RESHAPE_SHAPES: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = [
    # ==================== REGULAR TESTS ====================
    # Identity reshapes
    ((64, 64), (64, 64)),
    ((3, 32, 64), (3, 32, 64)),
    # 2D -> 2D reshapes
    ((64, 64), (32, 128)),
    ((128, 64), (64, 128)),
    ((32, 128), (64, 64)),
    # 2D -> 3D reshapes
    ((96, 64), (3, 32, 64)),
    ((128, 96), (4, 32, 96)),
    ((192, 64), (6, 32, 64)),
    # 2D -> 4D reshapes
    ((192, 64), (2, 3, 32, 64)),
    ((256, 96), (2, 4, 32, 96)),
    # 3D -> 2D reshapes
    ((3, 32, 64), (96, 64)),
    ((4, 64, 32), (256, 32)),
    ((5, 32, 64), (160, 64)),
    # 3D -> 3D reshapes
    ((2, 64, 64), (4, 32, 64)),
    ((3, 32, 96), (3, 96, 32)),
    ((6, 32, 64), (3, 64, 64)),
    # 3D -> 4D reshapes
    ((6, 32, 64), (2, 3, 32, 64)),
    ((12, 64, 32), (3, 4, 64, 32)),
    # 4D -> 2D reshapes
    ((2, 3, 32, 64), (192, 64)),
    ((2, 4, 64, 32), (512, 32)),
    # 4D -> 3D reshapes
    ((2, 3, 32, 64), (6, 32, 64)),
    ((2, 4, 32, 96), (8, 32, 96)),
    # 4D -> 4D reshapes
    ((2, 3, 32, 64), (3, 2, 32, 64)),
    ((2, 2, 64, 64), (4, 1, 64, 64)),
    # 5D -> 3D reshapes
    # Seems like it fails non deterministically
    # ((2, 3, 2, 32, 64), (12, 32, 64)),
    # 3D -> 5D reshapes
    ((12, 32, 64), (2, 3, 2, 32, 64)),
    # Inner dimension changes (more complex data movement)
    ((128, 32), (64, 64)),
    ((64, 128), (128, 64)),
    ((32, 192), (96, 64)),
    ((64, 96), (96, 64)),
    ((3, 64, 32), (3, 32, 64)),
    ((2, 128, 64), (2, 64, 128)),
    # ==================== WEIRD SHAPES ====================
    # Shapes with prime numbers and odd dimensions
    ((7, 7, 7), (49, 7)),
    ((49, 7), (7, 7, 7)),
    ((3, 11, 13), (33, 13)),
    ((33, 13), (3, 11, 13)),
    # Weird shapes with NOC issues
    pytest.param(((1, 32), (32, 1)), marks=NOC_ISSUE_SKIP),
    pytest.param(((2, 3, 5, 7), (6, 35)), marks=NOC_ISSUE_SKIP),
    pytest.param(((6, 35), (2, 3, 5, 7)), marks=NOC_ISSUE_SKIP),
    pytest.param(((11, 13, 2), (22, 13)), marks=NOC_ISSUE_SKIP),
    pytest.param(((22, 13), (11, 13, 2)), marks=NOC_ISSUE_SKIP),
    pytest.param(((3, 5, 7, 11), (15, 77)), marks=NOC_ISSUE_SKIP),
    pytest.param(((15, 77), (3, 5, 7, 11)), marks=NOC_ISSUE_SKIP),
    pytest.param(((2, 3, 5, 7, 11), (6, 5, 77)), marks=NOC_ISSUE_SKIP),
    pytest.param(((6, 5, 77), (2, 3, 5, 7, 11)), marks=NOC_ISSUE_SKIP),
    pytest.param(((5, 7, 9, 11), (35, 99)), marks=NOC_ISSUE_SKIP),
    pytest.param(((35, 99), (5, 7, 9, 11)), marks=NOC_ISSUE_SKIP),
    # 1D tensor shapes (not yet supported)
    pytest.param(((7, 11), (77,)), marks=ONE_D_SKIP),
    pytest.param(((77,), (7, 11)), marks=ONE_D_SKIP),
    pytest.param(((13, 17), (221,)), marks=ONE_D_SKIP),
    pytest.param(((221,), (13, 17)), marks=ONE_D_SKIP),
    pytest.param(((17, 19), (323,)), marks=ONE_D_SKIP),
    pytest.param(((323,), (17, 19)), marks=ONE_D_SKIP),
    pytest.param(((23, 29), (667,)), marks=ONE_D_SKIP),
    pytest.param(((667,), (23, 29)), marks=ONE_D_SKIP),
    # ==================== LLAMA 3.2 3B TESTS ====================
    pytest.param(((1024, 3072), (1, 1024, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((128256, 3072), (1, 128256, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((128,), (1, 128)), marks=ONE_D_SKIP),
    ((18, 128), (1, 18, 128)),
    ((18, 128), (1, 1, 18, 128)),
    pytest.param(((18,), (18, 1)), marks=ONE_D_SKIP),
    pytest.param(((18,), (1, 1, 18)), marks=ONE_D_SKIP),
    pytest.param(((1, 1024, 3072), (1024, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((1, 128256, 3072), (128256, 3072)), marks=SLOW_COMPILE_SKIP),
    ((1, 18, 128), (18, 128)),
    ((1, 18, 128), (1, 1, 18, 128)),
    pytest.param(((1, 1, 18), (18,)), marks=ONE_D_SKIP),
    pytest.param(((1, 1, 3072), (3072,)), marks=ONE_D_SKIP),
    pytest.param(((1, 1, 64), (1, 64, 1)), marks=NOC_ISSUE_SKIP),
    pytest.param(((1, 3072, 3072), (3072, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((1, 3072, 8192), (3072, 8192)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((1, 32, 18), (576,)), marks=ONE_D_SKIP),
    pytest.param(((1, 8192, 3072), (8192, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((3072, 3072), (1, 3072, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((3072, 8192), (1, 3072, 8192)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((3072,), (1, 1, 3072)), marks=ONE_D_SKIP),
    pytest.param(((32, 18, 128), (32, 1, 18, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 18, 1), (32, 18)), marks=NOC_ISSUE_SKIP),
    pytest.param(((32, 18, 24, 128), (576, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 18, 3072), (576, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 18, 8192), (576, 8192)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 18), (32, 18, 1)), marks=NOC_ISSUE_SKIP),
    ((32, 18), (1, 32, 18)),
    pytest.param(((32, 1, 18, 128), (32, 18, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 24, 128, 128), (768, 128, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 24, 18, 128), (768, 18, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 24, 18), (32, 24, 18, 1)), marks=NOC_ISSUE_SKIP),
    pytest.param(((32, 8, 128, 128), (32, 8, 1, 128, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 8, 3, 128, 128), (32, 24, 128, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((32, 8, 3, 128, 128), (768, 128, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((576, 1024), (32, 18, 8, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((576, 128256), (32, 18, 128256)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((576, 3072), (32, 18, 24, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((576, 3072), (32, 18, 3072)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((576, 8192), (32, 18, 8192)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((64,), (1, 1, 64)), marks=ONE_D_SKIP),
    pytest.param(((768, 18, 128), (32, 24, 18, 128)), marks=SLOW_COMPILE_SKIP),
    pytest.param(((8192, 3072), (1, 8192, 3072)), marks=SLOW_COMPILE_SKIP),
]


def shapes_to_id(shapes) -> str:
    """Generate a readable test ID from input/output shapes."""
    # Handle pytest.param objects
    if hasattr(shapes, "values"):
        input_shape, output_shape = shapes.values[0]
    else:
        input_shape, output_shape = shapes
    in_str = "x".join(str(d) for d in input_shape)
    out_str = "x".join(str(d) for d in output_shape)
    return f"{in_str}_to_{out_str}"


@pytest.mark.parametrize(
    "shapes", RESHAPE_SHAPES, ids=[shapes_to_id(s) for s in RESHAPE_SHAPES]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reshape(
    shapes: Tuple[Tuple[int, ...], Tuple[int, ...]],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Test reshape operation with various input/output shape combinations."""
    input_shape, output_shape = shapes

    def reshape_module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def reshape(in0, builder: TTIRBuilder, unit_attrs: List[str] = None):
            return builder.reshape(in0, output_shape, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        reshape_module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        test_base=request.node.name,
        module_dump=True,
        print_ir=False,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
