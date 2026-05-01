# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for tensor manipulation (TM) operations on the TTMetal backend.

Combines the original TM pipeline tests (permute, reshape, etc.) with
data-movement op tests (concat, slice, transpose, typecast, pad), the
collapse-tensors-to-2d pipeline tests, and the rearrange op tests.
"""

import itertools
import math
import pytest
import torch
import einops
from typing import Callable, List, Optional, Tuple

from ttmlir.dialects import ttir, ttcore
from ttmlir.ir import *

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import Marks, SkipIf, shape_str, shapes_list_str

pytestmark = pytest.mark.frontend("ttir")

# Skip reason for shapes with slow compilation due to inefficient coalescing factor calculation
SLOW_COMPILE_SKIP = pytest.mark.skip(
    reason="Slow compilation due to inefficient calculateCoalescingFactors, see https://github.com/tenstorrent/tt-mlir/issues/6375"
)

# Skip for NOC read issue
NOC_ISSUE_SKIP = pytest.mark.skip(
    reason="NOC read issue, see https://github.com/tenstorrent/tt-mlir/issues/6377"
)

# ============================================================
# TM pipeline tests (permute/reshape/broadcast etc.).
# ============================================================

# ==================== PERMUTE TESTS ====================


@pytest.mark.parametrize(
    "shape, permutation",
    [
        # 2d transpose
        pytest.param(
            (32, 128 * 500),
            [1, 0],
            marks=pytest.mark.skip_config(
                ["sim"],
                reason="L1 memory usage exceeds capacity on non-square grid (see #8079)",
            ),
        ),
        pytest.param(
            (32, 128 * 501),
            [1, 0],
            marks=[
                pytest.mark.skip_config(
                    ["n150"],
                    ["n300"],
                    reason="L1 memory usage exceeds capacity #7559",
                ),
                pytest.mark.skip_config(
                    ["p150"],
                    reason="L1 memory usage exceeds capacity on non-square grid (see #8079)",
                ),
            ],
        ),
        pytest.param(
            (32, 128 * 800),
            [1, 0],
            marks=[
                pytest.mark.skip_config(
                    ["n150"],
                    ["n300"],
                    reason="L1 memory usage exceeds capacity #7559",
                ),
                pytest.mark.skip_config(
                    ["p150"],
                    reason="L1 memory usage exceeds capacity on non-square grid (see #8079)",
                ),
            ],
        ),
        pytest.param(
            (32, 128 * 801),
            [1, 0],
            marks=pytest.mark.skip_config(
                ["n150"],
                ["n300"],
                ["p150"],
                ["p300"],
                reason="L1 memory usage exceeds capacity #7559",
            ),
        ),
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
            marks=[
                pytest.mark.skip_config(
                    ["p150"],
                    ["p300"],
                    reason="L1 memory usage exceeds capacity on p150/p300",
                ),
                pytest.mark.skip_config(
                    ["sim"],
                    reason="L1 memory usage exceeds capacity on non-square grid (see #8079)",
                ),
            ],
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
        pytest.param(
            (1, 96, 32, 128),
            [0, 1, 3, 2],
            marks=pytest.mark.skip_config(
                ["sim"],
                reason="L1 memory usage exceeds capacity on non-square grid (see #8079)",
            ),
        ),
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


@pytest.mark.parametrize("target", ["ttmetal"])
def test_permute_virtual_grid_1x64(target: str, request, device):
    """Test permute host transfers through a 1x64 virtual grid."""

    shape = (32, 2048)
    permutation = [1, 0]

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
    ((7, 7, 7), (49, 7)) | SkipIf("sim"),
    ((49, 7), (7, 7, 7)) | SkipIf("sim"),
    ((3, 11, 13), (33, 13)) | SkipIf("sim"),
    ((33, 13), (3, 11, 13)) | SkipIf("sim"),
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
    # ==================== RESHAPE + PERMUTE (TRAILING 1 FLIP) TESTS ====================
    # Qwen3 32B
    ((1, 64), (1, 64, 1)),
    # GPT OSS 120B
    ((1, 128), (128, 1)),
    ((1, 16), (1, 16, 1, 1)) | SkipIf("sim"),
    # Kimi K2 1T
    ((1, 32), (32, 1)),
    ((32,), (32, 1)),
    # DeepSeek 671B
    ((1, 32), (1, 32, 1)),
    ((8,), (8, 1, 1)) | SkipIf("sim"),
    # GLM 358B
    ((1, 32, 8), (1, 32, 8, 1)),
    ((1, 96, 32), (1, 96, 32, 1)),
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
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.int32 | SkipIf("sim"),
        torch.int64 | SkipIf("sim"),
        torch.bool,
    ],
    ids=["f32", "bf16", "i32", "i64", "i1"],
)
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
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
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

    if dtype == torch.int32:
        pytest.xfail(
            reason="Currently no llk for multiplying a tile with a scalar for i32, Issue: https://github.com/tenstorrent/tt-mlir/issues/7946"
        )

    num_elements = shape[0] * shape[1]
    end = start + num_elements * step
    arange_dimension = 1  # Arange is always on the last dimension

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
        atol=1e-6,
        check_atol=True,
    )


# ==================== EMBEDDING TESTS ====================


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "indices_shape, weight_shape",
    [
        ((1, 32), (1024, 32)),
        ((1, 64), (1024, 32)),
        ((1, 32), (1024, 64)),
        ((1, 32), (2048, 32)),
        ((2, 16), (1024, 32)),
        ((4, 8), (128, 16)),
        ((1, 128), (40960, 128)),
    ],
    ids=lambda shape: shape_str(shape),
)
def test_embedding(
    target: str, request, device, indices_shape: Shape, weight_shape: Shape
):
    num_indices = math.prod(indices_shape)

    def embedding_module(builder: TTIRBuilder):
        @builder.func([indices_shape, weight_shape], [torch.int32, torch.float32])
        def embedding(
            indices: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            valid_indices = (
                torch.arange(num_indices, dtype=torch.int32).reshape(indices_shape) * 97
            ) % weight_shape[0]
            builder.set_goldens(inputs={indices: valid_indices})
            return builder.embedding(indices, weight, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        embedding_module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


_EMBEDDING_DTYPE_IDS = {
    torch.int32: "i32",
    torch.uint32: "ui32",
    torch.float32: "f32",
    torch.bfloat16: "bf16",
}


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "indices_dtype, weight_dtype",
    [
        pytest.param(
            indices_dtype,
            weight_dtype,
            id=f"{_EMBEDDING_DTYPE_IDS[indices_dtype]}_indices_"
            f"{_EMBEDDING_DTYPE_IDS[weight_dtype]}_table",
        )
        for indices_dtype, weight_dtype in itertools.product(
            [torch.int32, torch.uint32],
            [torch.float32, torch.bfloat16, torch.int32],
        )
    ],
)
def test_embedding_supported_dtypes(
    target: str,
    request,
    device,
    indices_dtype: torch.dtype,
    weight_dtype: torch.dtype,
):
    indices_shape = (2, 3)
    weight_shape = (16, 5)
    num_indices = math.prod(indices_shape)

    def embedding_module(builder: TTIRBuilder):
        @builder.func([indices_shape, weight_shape], [indices_dtype, weight_dtype])
        def embedding(
            indices: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            valid_indices = (
                (
                    torch.arange(num_indices, dtype=torch.int64).reshape(indices_shape)
                    * 7
                )
                % weight_shape[0]
            ).to(indices_dtype)
            builder.set_goldens(inputs={indices: valid_indices})
            return builder.embedding(indices, weight, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        embedding_module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


# ============================================================
# Collapse-tensors-to-2d pipeline tests
# (previously test_tensor_collapsing.py).
# ============================================================


def module_elementwise_add_3d_add(builder: TTIRBuilder):
    @builder.func([(3, 32, 64), (3, 32, 64)], [torch.float32, torch.float32])
    def elementwise_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
        """Element-wise addition operation."""
        return builder.add(in0, in1)


def module_elementwise_add_4d_add(builder: TTIRBuilder):
    @builder.func([(2, 3, 64, 32), (2, 3, 64, 32)], [torch.float32, torch.float32])
    def elementwise_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
        """Element-wise addition operation."""
        return builder.add(in0, in1)


def module_batch_matmul(builder: TTIRBuilder):
    @builder.func([(2, 32, 64), (2, 64, 32)], [torch.float32, torch.float32])
    def batch_matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
        """Batch matrix multiplication operation."""
        return builder.matmul(in0, in1)


def module_elementwise_multiply_3d_multiply(builder: TTIRBuilder):
    @builder.func([(3, 32, 64), (3, 32, 64)], [torch.float32, torch.float32])
    def elementwise_multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
        """Element-wise multiplication operation."""
        return builder.multiply(in0, in1)


def module_unary_exp_2d_exp(builder: TTIRBuilder):
    @builder.func([(3, 32, 64)], [torch.float32])
    def unary_exp(in0: Operand, builder: TTIRBuilder):
        """Unary exponential operation."""
        return builder.exp(in0)


def module_unary_exp_4d_exp(builder: TTIRBuilder):
    @builder.func([(1, 2, 32, 32)], [torch.float32])
    def unary_exp(in0: Operand, builder: TTIRBuilder):
        """Unary exponential operation."""
        return builder.exp(in0)


def module_transpose_inner_dims(builder: TTIRBuilder):
    @builder.func([(3, 32, 64)], [torch.float32])
    def transpose_inner_dims(in0: Operand, builder: TTIRBuilder):
        """Transpose operation on inner dimensions (last two dims)."""
        return builder.transpose(in0, 1, 2)


@pytest.mark.parametrize(
    "test_func,test_name",
    [
        # 3D element-wise operations (working with non-collapsed tensors)
        (module_elementwise_add_3d_add, "3d_add"),
        (module_elementwise_multiply_3d_multiply, "3d_multiply"),
        (module_unary_exp_2d_exp, "3d_exp"),
        # 4D element-wise operations (working with non-collapsed tensors)
        pytest.param(module_elementwise_add_4d_add, "4d_add"),
        (module_unary_exp_4d_exp, "4d_exp"),
        # Batched matmul (fixed in #6648)
        (module_batch_matmul, "matmul"),
        # Operations with known issues (marked as skip)
        pytest.param(
            module_transpose_inner_dims,
            "transpose",
            marks=pytest.mark.skip(
                reason="Hardcoded rank==2 assertions in permute rewriter cause core dump"
            ),
        ),
    ],
    ids=["3d_add", "3d_multiply", "3d_exp", "4d_add", "4d_exp", "matmul", "transpose"],
)
@pytest.mark.parametrize(
    "collapse_tensors", [True, False], ids=["collapsed", "non_collapsed"]
)
@pytest.mark.parametrize("target", ["ttmetal"], ids=["ttmetal"])
def test_uncollapsed_tensors(
    test_func,
    test_name: str,
    collapse_tensors: bool,
    target: str,
    request,
    device,
):
    """Test tensor operations with and without tensor collapsing to 2D."""

    pipeline_options = f"{{collapse-tensors-2d={str(collapse_tensors).lower()}}}"
    pipeline = f"ttir-to-ttmetal-pipeline{pipeline_options}"

    compile_and_execute_ttir(
        test_func,
        target=target,
        custom_pipeline=pipeline,
        test_base=f"{request.node.name}_{test_name}_{'collapsed' if collapse_tensors else 'non_collapsed'}",
        device=device,
    )


# ============================================================
# Rearrange op tests (previously test_rearrange.py).
# ============================================================


def _test_pattern_map(pattern, shape, pattern_map):
    print(pattern, ":", shape, ":", pattern_map)
    t = torch.randn(shape)
    golden = einops.rearrange(t, pattern)
    output = torch.zeros(golden.shape)
    for pos in itertools.product(*[range(dim) for dim in output.shape]):
        p = ttir.ir.affine_map_compose(pattern_map, pos)
        output[slice(*pos)] = t[slice(*p)]
    assert torch.allclose(output, golden)


# Currently any permute that involves the innermost dim is not supported
@pytest.mark.parametrize(
    "shape,pattern",
    [
        # Tilized: inner dims preserved
        ((3, 32, 64), "z y x -> z y x"),
        ((2, 3, 32, 64), "w z y x -> z w y x"),
        ((2, 3, 32, 64), "w z y x -> (w z) y x"),
        ((4, 3, 32, 64), "w z y x -> (z w) y x"),
        # Non-tilized: inner dims modified
        ((3, 32, 32), "z y x -> y z x"),
        ((3, 32, 32), "z y x -> (y z) x"),
        ((3, 32, 32), "z y x -> y (z x)"),
        # Unaligned
        ((3, 4, 5), "z y x -> y z x"),
        ((5, 7, 8), "z y x -> (y z) x"),
        ((5, 7, 8), "z y x -> y (z x)"),
        # Multicore
        ((2, 4, 250), "z y x -> y z x"),
        ((2, 7, 180), "z y x -> (y z) x"),
        ((25, 7, 8), "z y x -> y (z x)"),
        ((50, 7, 8), "z y x -> y (z x)"),
        ((50, 15, 8), "z y x -> z (y x)"),
        ((4, 85, 1055), "z y x -> (z y) x"),
        # 4d
        ((2, 3, 4, 32), "w z y x -> y w z x"),
        ((2, 3, 4, 32), "w z y x -> y (w z) x"),
        ((2, 3, 4, 32), "w z y x -> (y w z) x"),
        ((2, 3, 4, 32), "w z y x -> (y w) z x"),
        ((2, 3, 4, 32), "w z y x -> (y w) (z x)"),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_rearrange(
    shape,
    pattern,
    target: str,
    request,
    device,
):
    # Enable for local debug of the pattern -> affine map conversion.
    test_pattern_map = False
    if test_pattern_map:
        pattern_map = ttir.ir.rearrange_inv_pattern_map(Context(), pattern, shape)
        _test_pattern_map(pattern, shape, pattern_map)

    def rearrange_module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def rearrange(in0, builder: TTIRBuilder, unit_attrs: List[str] = None):
            return builder.rearrange(in0, pattern, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        rearrange_module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


# ============================================================
# Data-movement op mirrors (concat/slice/transpose/typecast/pad,
# previously test_data_movement.py).
# ============================================================

# Concat tests
@pytest.mark.parametrize(
    "shapes,dim",
    [
        ##################################
        #               2D               #
        ##################################
        # Trivial aligned inputs
        ([(32, 32), (32, 32)], 1),
        ([(32, 32), (32, 32)], 0),
        # Larger aligned inputs
        ([(64, 96), (64, 1024)], 1),
        ([(96, 64), (1024, 64)], 0),
        ([(256, 256), (256, 256)], 1),
        ([(256, 256), (256, 256)], 0),
        # Unaligned in the non-concat dim
        ([(7, 64), (7, 96)], 1),
        ([(96, 3), (64, 3)], 0),
        # 3-concat, last input unaligned in the concat dim
        ([(128, 64), (128, 32), (128, 16)], 1),
        ([(64, 128), (32, 128), (16, 128)], 0),
        # 4-concat
        ([(64, 64), (64, 32), (64, 128), (64, 96)], 1),
        ([(64, 64), (32, 64), (128, 64), (96, 64)], 0),
        # Max-concat (limited by max 32 CBs on WH)
        ([(64, 32)] * 31, 1),
        ([(32, 64)] * 31, 0),
        ##################################
        #               3D               #
        ##################################
        ([(17, 32, 64), (17, 32, 96)], 2),
        ([(19, 64, 32), (19, 96, 32)], 1),
        ([(11, 64, 64), (13, 64, 64)], 0),
        ([(10, 64, 32), (10, 64, 64), (10, 64, 96)], 2),
        ([(10, 96, 64), (10, 32, 64), (10, 64, 64)], 1),
        ([(11, 64, 64), (12, 64, 64), (13, 64, 64)], 0),
        ##################################
        #               4D               #
        ##################################
        ([(3, 7, 32, 64), (3, 7, 32, 96)], 3),
        ([(7, 3, 64, 64), (7, 3, 32, 64)], 2),
        ([(2, 6, 64, 32), (2, 9, 64, 32)], 1),
        ([(8, 5, 32, 64), (3, 5, 32, 64)], 0),
        ([(2, 3, 64, 32), (2, 3, 64, 64), (2, 3, 64, 96)], 3),
        ([(3, 2, 96, 64), (3, 2, 32, 64), (3, 2, 64, 64)], 2),
        ([(4, 2, 64, 64), (4, 1, 64, 64), (4, 3, 64, 64)], 1),
        ([(3, 3, 64, 64), (2, 3, 64, 64), (1, 3, 64, 64)], 0),
        ##################################
        #    Large tensors (multi-core)  #
        ##################################
        ([(1, 32, 128, 64), (1, 32, 128, 64)], 3),
        ([(1, 32, 128, 128), (1, 32, 128, 128)], 3),
        ([(1, 32, 64, 128), (1, 32, 64, 128)], 2),
        ([(512, 512), (512, 512)], 1),
        ([(512, 512), (512, 512)], 0),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_concat(shapes: List[Shape], dim: int, target: str, request, device):
    def module(builder: TTIRBuilder):
        dtypes = [torch.float32] * len(shapes)

        @builder.func(shapes, dtypes)
        def concat_wrapper(
            *args,
            unit_attrs: Optional[List[str]] = None,
        ):
            inputs = args[:-1]
            builder = args[-1]
            return builder.concat(list(inputs), dim, unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Slice tests
@pytest.mark.parametrize(
    "shape,begins,ends,step",
    [
        # Tilized: inner dims preserved
        ((4, 32, 64), [1, 0, 0], [3, 32, 64], [1, 1, 1]),
        ((6, 32, 64), [0, 0, 0], [4, 32, 64], [2, 1, 1]),
        ((4, 5, 32, 64), [1, 2, 0, 0], [3, 4, 32, 64], [1, 1, 1, 1]),
        ((8, 6, 64, 32), [2, 1, 0, 0], [6, 5, 64, 32], [1, 1, 1, 1]),
        ((2, 4, 3, 32, 64), [0, 1, 1, 0, 0], [2, 3, 2, 32, 64], [1, 1, 1, 1, 1]),
        # Simple 1D
        ((64,), [0], [32], None),
        # Strided 1D
        ((70,), [3], [62], [7]),
        ((2048,), [0], [2048], [3]),
        # Simple 2D
        ((64, 64), [0, 0], [32, 32], None),
        # Crop 2D
        ((64, 64), [10, 20], [50, 60], [1, 1]),
        # Every three rows/cols
        ((192, 64), [2, 0], [192, 64], [3, 1]),
        ((64, 192), [0, 2], [64, 192], [1, 3]),
        # Sample large 2D tensors
        pytest.param(
            (32, 131072),
            [0, 3],
            [32, 128 * 991],
            [2, 991],
            marks=pytest.mark.skip_config(
                ["ttmetal", "p150"],
                ["ttmetal", "p300"],
                reason="L1 memory usage exceeds capacity #7559",
            ),
        ),
        ((131072, 32), [5, 1], [128 * 997, 32], [997, 2]),
        ((1024, 1024), [3, 2], [64 * 11, 64 * 13], [11, 13]),
        # Simple 3D
        ((2, 64, 32), [0, 0, 0], [1, 64, 32], None),
        # Crop 3D
        ((19, 160, 64), [0, 0, 0], [19, 96, 32], None),
        # Interleaved 3D
        ((2, 64, 32), [0, 1, 0], [1, 64, 32], [1, 2, 1]),
        ((2, 64, 32), [0, 1, 0], [1, 64, 32], [1, 2, 2]),
        # Strided crop 3D
        ((64, 64, 64), [10, 20, 28], [50, 60, 64], [2, 2, 1]),
        ((64, 64, 64), [10, 20, 30], [50, 60, 64], [2, 2, 1]),
        ((64, 64, 64), [5, 30, 12], [11, 34, 36], [3, 1, 4]),
        # Minus 1
        ((3, 512, 256), [0, 1, 0], [3, 512, 255], None),
        ((5, 65, 1025), [0, 0, 1], [5, 64, 1025], None),
        # Simple 4D
        ((2, 24, 32, 128), [1, 8, 3, 64], [2, 16, 7, 128], None),
        # NCHW: 2nd half - green - down sample & make square
        ((4, 3, 64, 96), [2, 1, 1, 0], [4, 2, 64, 96], [1, 1, 2, 3]),
        # NHWC: odd - crop - blue & alpha
        ((6, 64, 64, 4), [1, 15, 16, 2], [6, 47, 48, 4], [2, 1, 1, 1]),
        # Simple 5D
        ((2, 4, 6, 64, 64), [0, 1, 0, 0, 0], [1, 2, 1, 32, 32], None),
        # Mixed 5D
        ((3, 4, 5, 128, 128), [1, 0, 3, 32, 64], [3, 4, 4, 96, 128], [1, 2, 1, 1, 1]),
        # Pick a single element
        ((3, 20, 14, 64, 64), [1, 5, 6, 31, 32], [2, 6, 7, 32, 33], None),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_slice(
    shape: Shape,
    begins: List[int],
    ends: List[int],
    step: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def slice_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.slice(in0, begins, ends, step, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Transpose tests
@pytest.mark.parametrize("shape", [(64, 32)], ids=shape_str)
@pytest.mark.parametrize("transpose_dims", [(0, 1)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_transpose(
    shape: Shape, transpose_dims: List[int], target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def transpose_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.transpose(
                in0,
                dim0=transpose_dims[0],
                dim1=transpose_dims[1],
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Typecast tests
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "from_type,to_type",
    [
        pytest.param(
            torch.int32, torch.float32, marks=pytest.mark.skip_config(["sim"])
        ),
        (torch.float32, torch.int32),
        (torch.bfloat16, torch.float32),
        pytest.param(
            torch.float32, torch.bfloat16, marks=pytest.mark.skip_config(["sim"])
        ),
    ],
    ids=["i32-f32", "f32-i32", "bf16-f32", "f32-bf16"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_typecast(
    shape: Shape,
    from_type: torch.dtype,
    to_type: torch.dtype,
    target: str,
    request,
    device,
):
    if from_type == torch.float32 and to_type == torch.int32:
        pytest.xfail("ttmetal does not support float32 to int32 typecast")

    def module(builder: TTIRBuilder):
        @builder.func([shape], [from_type])
        def typecast(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.typecast(in0, output_type=to_type, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        pipeline_options=[],
    )
