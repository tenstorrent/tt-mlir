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
from conftest import get_request_kwargs

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")

# Skip reason for shapes with slow compilation due to inefficient coalescing factor calculation
SLOW_COMPILE_SKIP = pytest.mark.skip(
    reason="Slow compilation due to inefficient calculateCoalescingFactors, see https://github.com/tenstorrent/tt-mlir/issues/6375"
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
        # 4d outer permutes (tilized)
        [(1, 18, 8, 128), [1, 0, 2, 3]],
        # 5d outer permutes
        [(1, 3, 3, 3, 3), [0, 2, 1, 3, 4]],
        [(1, 3, 3, 3, 3), [0, 2, 1, 3, 4]],
        [(5, 7, 2, 3, 3), [0, 2, 1, 3, 4]],
        # # 4d inner, then outer permute
        [(3, 32, 32, 32), [0, 3, 1, 2]],
        # # 5d inner, then outer permute
        [(1, 3, 3, 3, 3), [0, 2, 1, 3, 4]],
        # 3d outer permutes (gpt-20b)
        [(1, 32, 128), [0, 2, 1]],
        # 4d outer permutes (gpt-20b)
        [(1, 128, 8, 64), [0, 2, 1, 3]],
        [(1, 128, 1, 64), [0, 2, 1, 3]],
        # 4d inner permutes (gpt-20b)
        [(1, 8, 128, 64), [0, 1, 3, 2]],
        [(1, 8, 64, 128), [0, 1, 3, 2]],
        # 4d inner permutes (llama3-70b, qwen3-32b)
        pytest.param(
            (32, 8, 128, 128),
            [0, 1, 3, 2],
            marks=pytest.mark.skip_config(
                ["p150"],
                ["p300"],
                reason="L1 memory usage exceeds capacity on p150/p300",
            ),
        ),
        # 4d complex permutes (llama3-70b, qwen3-32b)
        [(32, 1, 1, 128), [2, 1, 0, 3]],
        # 4d outer permutes (deepseek-671b)
        [(1, 16, 32, 32), [0, 2, 1, 3]],
        [(1, 16, 32, 128), [0, 2, 1, 3]],
        # 3d inner permutes (glm-358b)
        [(1, 32, 32), [0, 2, 1]],
        # 4d outer permutes (glm-358b)
        [(1, 32, 8, 128), [0, 2, 1, 3]],
        [(1, 32, 96, 128), [0, 2, 1, 3]],
        # 4d inner permutes (glm-358b)
        [(1, 96, 32, 128), [0, 1, 3, 2]],
        # 3d inner permutes (gpt_oss-120b)
        [(1, 128, 32), [0, 2, 1]],
        # 4d outer permutes (gpt_oss-120b)
        [(1, 128, 16, 64), [0, 2, 1, 3]],
        [(1, 128, 2, 64), [0, 2, 1, 3]],
        # 4d inner permutes (gpt_oss-120b)
        [(1, 16, 128, 64), [0, 1, 3, 2]],
        [(1, 2, 128, 64), [0, 1, 3, 2]],
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
        **get_request_kwargs(request),
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '}}}",
    )


# ==================== CONCATENATE HEADS TESTS ====================


@pytest.mark.parametrize(
    "input_shape",
    [
        # Format: (batch, num_heads, seq_len, head_dim)
        (1, 8, 32, 64),
        (1, 12, 64, 64),
        (1, 16, 32, 128),
        (2, 8, 32, 64),
        (1, 24, 32, 128),
        (2, 24, 32, 128),
        (1, 32, 64, 128),
        (1, 8, 128, 64),
        (1, 4, 32, 32),
        (1, 2, 32, 64),
        (1, 12, 256, 64),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_concatenate_heads(
    input_shape: Tuple[int, int, int, int], target: str, request, device
):
    """Test concatenate_heads operation on TTMetal backend.

    Concatenate heads transforms:
    Input: [batch, num_heads, seq_len, head_dim]
    Output: [batch, seq_len, num_heads * head_dim]
    """
    batch, num_heads, seq_len, head_dim = input_shape
    output_shape = (batch, seq_len, num_heads * head_dim)

    def concatenate_heads_module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
        def concatenate_heads(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            return builder.concatenate_heads(in0, output_type=torch.float32)

    compile_and_execute_ttir(
        concatenate_heads_module,
        target=target,
        device=device,
        print_ir=True,
        **get_request_kwargs(request),
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
    # 2D -> 3D reshapes
    ((96, 64), (3, 32, 64)),
    ((128, 96), (4, 32, 96)),
    # 2D -> 4D reshapes
    ((192, 64), (2, 3, 32, 64)),
    ((256, 96), (2, 4, 32, 96)),
    # 3D -> 2D reshapes
    ((3, 32, 64), (96, 64)),
    ((5, 32, 64), (160, 64)),
    # 3D -> 3D reshapes
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
    # 1D tensor shapes
    ((1,), (1, 1, 1)),
    ((1,), (1, 1, 1, 1)),
    ((128,), (1, 128)),
    ((1, 64), (64,)),
    ((1, 1, 128), (128,)),
    ((128,), (2, 64)),
    ((2, 64), (128,)),
    ((64,), (64,)),
    ((128,), (1, 1, 1, 128)),
    # LLAMA 3.2 3B TESTS
    ((18, 128), (1, 18, 128)),
    ((18, 128), (1, 1, 18, 128)),
    ((1, 18, 128), (18, 128)),
    ((1, 18, 128), (1, 1, 18, 128)),
    ((32, 18), (1, 32, 18)),
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
        **get_request_kwargs(request),
    )


# ==================== ARANGE TESTS ====================


@pytest.mark.parametrize(
    "shape,start,step",
    [
        ((1, 32), 0, 1),  # Single tile
        ((1, 64), 32, 2),  # Two tiles
        ((1, 96), 64, 1),  # Three tiles
        ((1, 128), 0, 1),  # Four tiles (from GPT model)
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_arange(
    shape: tuple,
    start: int,
    step: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Test arange operation on TTMetal backend.

    Tests tiled arange implementation with various shapes and parameters.
    """
    num_elements = shape[0] * shape[1]
    end = start + num_elements * step
    arange_dimension = 1  # Arange is always on the last dimension

    golden = torch.arange(start, end, step, dtype=dtype).reshape(shape)

    def arange_module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def arange(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            result = builder.arange(
                shape=list(shape),
                dtype=dtype,
                start=start,
                end=end,
                step=step,
                arange_dimension=arange_dimension,
                unit_attrs=unit_attrs,
            )
            return result

    compile_and_execute_ttir(
        arange_module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )
