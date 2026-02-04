# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for reshape + compute operation patterns on TTMetal backend.

These test cases are derived from real model IR to exercise reshape operations
composed with compute ops (eltwise binary/unary) as they appear in production
models:
  - GPT-OSS 120B
  - Kimik2 1T
  - GLM 358B
  - DeepSeek 671B
  - Qwen3 32B

Common patterns tested:
  1. LayerNorm: reshape [1, N] -> [N, 1] followed by broadcast multiply
  2. Rotary embedding: reshape to insert/remove unit dims, then cos/sin/multiply
  3. Attention: reshape to flatten/unflatten head dims, then add
  4. RMSNorm: reshape then rsqrt then multiply
"""

import pytest
import torch
from typing import List, Tuple

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


# ==================== HELPER ====================


def _id(name: str, in_shape, out_shape, op: str) -> str:
    """Build a readable test ID: model__INxIN_to_OUTxOUT__op."""
    in_str = "x".join(str(d) for d in in_shape)
    out_str = "x".join(str(d) for d in out_shape)
    return f"{name}__{in_str}_to_{out_str}__{op}"


# ==================== TEST CASES ====================
#
# Each entry is:
#   (model_tag, input_shape, reshape_shape, compute_op, compute_extra_input_shape)
#
# For unary ops compute_extra_input_shape is None.
# For binary ops compute_extra_input_shape is the shape of the second operand
# (after broadcast, i.e. the shape that both operands share after reshape).

# ---------- reshape -> unary compute ----------

RESHAPE_UNARY_CASES: List = [
    # GLM-358B: reshape 3D -> 4D, then rsqrt (RMSNorm pattern)
    # %58 = reshape(%57) : (1x32x8xf32) -> (1x32x8x1xf32)
    # %59 = rsqrt(%58)
    pytest.param(
        (1, 32, 8),
        (1, 32, 8, 1),
        "rsqrt",
        id=_id("glm358b", (1, 32, 8), (1, 32, 8, 1), "rsqrt"),
    ),
    # GLM-358B: reshape 3D -> 4D, then rsqrt (RMSNorm for MHA heads)
    # %105 = reshape(%104) : (1x32x96xf32) -> (1x32x96x1xf32)
    # %106 = rsqrt(%105)
    pytest.param(
        (1, 32, 96),
        (1, 32, 96, 1),
        "rsqrt",
        id=_id("glm358b", (1, 32, 96), (1, 32, 96, 1), "rsqrt"),
    ),
    # DeepSeek-671B: reshape 2D -> 3D, then rsqrt (RMSNorm)
    # %136 = reshape(%135) : (1x32xf32) -> (1x32x1xf32)
    # %137 = rsqrt(%136)
    pytest.param(
        (1, 32),
        (1, 32, 1),
        "rsqrt",
        id=_id("deepseek671b", (1, 32), (1, 32, 1), "rsqrt"),
    ),
    # Qwen3-32B: reshape 2D -> 2D (flatten batch*seq), then rsqrt
    # %867 = reshape(%866) : (32x17xf32) -> (544x1xf32)
    # %868 = rsqrt(%867)
    pytest.param(
        (32, 17),
        (544, 1),
        "rsqrt",
        id=_id("qwen3_32b", (32, 17), (544, 1), "rsqrt"),
    ),
    # GPT-OSS-120B: reshape 3D -> 4D, then cos (rotary embedding)
    # %596 = reshape(%595) : (1x128x32xf32) -> (1x1x128x32xf32)
    # %597 = cos(%596)
    pytest.param(
        (1, 128, 32),
        (1, 1, 128, 32),
        "cos",
        id=_id("gpt_oss120b", (1, 128, 32), (1, 1, 128, 32), "cos"),
    ),
    # GPT-OSS-120B: reshape 3D -> 4D, then sin (rotary embedding)
    # %603 = reshape(%595) : (1x128x32xf32) -> (1x1x128x32xf32)
    # %604 = sin(%603)
    pytest.param(
        (1, 128, 32),
        (1, 1, 128, 32),
        "sin",
        id=_id("gpt_oss120b", (1, 128, 32), (1, 1, 128, 32), "sin"),
    ),
    # GLM-358B: reshape 3D -> 4D, then cos (rotary embedding)
    # %71 = reshape(%70) : (1x32x32xf32) -> (1x1x32x32xf32)
    # cos of concat of two such reshapes -> 1x1x32x64
    pytest.param(
        (1, 32, 64),
        (1, 1, 32, 64),
        "cos",
        id=_id("glm358b", (1, 32, 64), (1, 1, 32, 64), "cos"),
    ),
    # Qwen3-32B: reshape 3D -> 4D, then cos (rotary embedding)
    # %894 = reshape(%893) : (1x17x64xf32) -> (1x1x17x64xf32)
    # cos of concat -> 1x1x17x128
    pytest.param(
        (1, 17, 128),
        (1, 1, 17, 128),
        "cos",
        id=_id("qwen3_32b", (1, 17, 128), (1, 1, 17, 128), "cos"),
    ),
    # Qwen3-32B: same but sin
    pytest.param(
        (1, 17, 128),
        (1, 1, 17, 128),
        "sin",
        id=_id("qwen3_32b", (1, 17, 128), (1, 1, 17, 128), "sin"),
    ),
]


@pytest.mark.parametrize("input_shape, reshape_shape, op_name", RESHAPE_UNARY_CASES)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reshape_then_unary(
    input_shape: Shape,
    reshape_shape: Tuple[int, ...],
    op_name: str,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Reshape followed by a unary compute op (patterns from real models)."""

    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def reshape_unary(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            reshaped = builder.reshape(in0, list(reshape_shape))
            op_fn = getattr(builder, op_name)
            return op_fn(reshaped)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


# ---------- reshape -> binary compute ----------

RESHAPE_BINARY_CASES: List = [
    # ---- LayerNorm patterns: reshape [1,N] -> [N,1] then multiply ----
    # Kimik2-1T: reshape (1x32) -> (32x1), multiply with (32x7168)
    # %55 = reshape(%54) : (1x32xf32) -> (32x1xf32)
    # %57 = multiply(%48, %56)  -- after broadcast of %55 to 32x7168
    pytest.param(
        (1, 32),
        (32, 1),
        "multiply",
        (32, 64),
        id=_id("kimik2_1t", (1, 32), (32, 1), "multiply") + "__32x64",
    ),
    # Kimik2-1T: reshape (1x32) -> (32x1), multiply with (32x1536)
    pytest.param(
        (1, 32),
        (32, 1),
        "multiply",
        (32, 96),
        id=_id("kimik2_1t", (1, 32), (32, 1), "multiply") + "__32x96",
    ),
    # DeepSeek-671B: reshape (1x32) -> (32x1), multiply with (32x2048)
    # %122 = reshape(%121) : (1x32xf32) -> (32x1xf32)
    # %124 = multiply(%115, %123)
    pytest.param(
        (1, 32),
        (32, 1),
        "multiply",
        (32, 128),
        id=_id("deepseek671b", (1, 32), (32, 1), "multiply") + "__32x128",
    ),
    # GLM-358B: reshape (1x32) -> (32x1), multiply with (32x5120)
    # %29 = reshape(%28) : (1x32xf32) -> (32x1xf32)
    # %31 = multiply(%22, %30)
    pytest.param(
        (1, 32),
        (32, 1),
        "multiply",
        (32, 160),
        id=_id("glm358b", (1, 32), (32, 1), "multiply") + "__32x160",
    ),
    # GPT-OSS-120B: reshape (1x128) -> (128x1), multiply with (128x360)
    # %576 = reshape(%575) : (1x128xf32) -> (128x1xf32)
    # %578 = multiply(%568, %577)
    pytest.param(
        (1, 128),
        (128, 1),
        "multiply",
        (128, 64),
        id=_id("gpt_oss120b", (1, 128), (128, 1), "multiply") + "__128x64",
    ),
    # Qwen3-32B: reshape (32x17) -> (544x1), multiply with (544x5120) (large)
    # Use smaller proxy shape to keep test fast
    pytest.param(
        (32, 17),
        (544, 1),
        "multiply",
        (544, 64),
        id=_id("qwen3_32b", (32, 17), (544, 1), "multiply") + "__544x64",
    ),
    # ---- Attention reshape + add patterns ----
    # GPT-OSS-120B: reshape 4D -> 2D, then add (attention bias add)
    # %588 = reshape(%587) : (1x128x16x64xbf16) -> (128x1024xbf16)
    # %589 = add(%583, %588)
    pytest.param(
        (1, 128, 2, 64),
        (128, 128),
        "add",
        (128, 128),
        id=_id("gpt_oss120b", (1, 128, 2, 64), (128, 128), "add"),
    ),
    # GLM-358B: reshape 4D -> 2D, then add (attention bias add)
    # %40 = reshape(%39) : (1x32x8x128xbf16) -> (32x1024xbf16)
    # %41 = add(%35, %40)
    pytest.param(
        (1, 32, 8, 128),
        (32, 1024),
        "add",
        (32, 1024),
        id=_id("glm358b", (1, 32, 8, 128), (32, 1024), "add"),
    ),
    # Kimik2-1T: reshape 3D -> 2D, then multiply (RMSNorm weight mul)
    # %149 = reshape(%141) : (1x32x512xf32) -> (32x512xf32)
    # %150 = multiply(%149, %148)
    pytest.param(
        (1, 32, 512),
        (32, 512),
        "multiply",
        (32, 512),
        id=_id("kimik2_1t", (1, 32, 512), (32, 512), "multiply"),
    ),
    # ---- Rotary embedding reshape + multiply patterns ----
    # GPT-OSS-120B: reshape 3D -> 4D, cos, then multiply (RoPE)
    # %596 = reshape(%595) : (1x128x32xf32) -> (1x1x128x32xf32)
    # %597 = cos(%596)
    # %598 = multiply(%597, %5)
    pytest.param(
        (1, 1, 128, 32),
        (1, 1, 128, 32),
        "multiply",
        (1, 1, 128, 32),
        id=_id("gpt_oss120b", (1, 1, 128, 32), (1, 1, 128, 32), "multiply") + "__rope",
    ),
    # ---- Reshape + subtract patterns ----
    # GPT-OSS-120B: reshape -> multiply then subtract (RoPE sin/cos combine)
    # %634 = subtract(%629, %633) on 5D tensors after reshape
    pytest.param(
        (1, 2, 1, 128, 32),
        (1, 2, 1, 128, 32),
        "subtract",
        (1, 2, 1, 128, 32),
        id=_id(
            "gpt_oss120b",
            (1, 2, 1, 128, 32),
            (1, 2, 1, 128, 32),
            "subtract",
        )
        + "__rope",
    ),
]


@pytest.mark.parametrize(
    "input_shape, reshape_shape, op_name, rhs_shape", RESHAPE_BINARY_CASES
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reshape_then_binary(
    input_shape: Shape,
    reshape_shape: Tuple[int, ...],
    op_name: str,
    rhs_shape: Tuple[int, ...],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Reshape one input, then apply a binary op with another input."""

    def module(builder: TTIRBuilder):
        @builder.func([input_shape, rhs_shape], [dtype, dtype])
        def reshape_binary(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            reshaped = builder.reshape(in0, list(reshape_shape))
            op_fn = getattr(builder, op_name)
            return op_fn(reshaped, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


# ---------- reshape -> unary -> binary (two-step patterns) ----------

RESHAPE_UNARY_BINARY_CASES: List = [
    # GLM-358B: reshape -> rsqrt -> multiply (full RMSNorm tail)
    # %58 = reshape : (1x32x8xf32) -> (1x32x8x1xf32)
    # %59 = rsqrt(%58)
    # broadcast -> multiply
    pytest.param(
        (1, 32, 8),
        (1, 32, 8, 1),
        "rsqrt",
        "multiply",
        (1, 32, 8, 1),
        id=_id("glm358b", (1, 32, 8), (1, 32, 8, 1), "rsqrt_multiply"),
    ),
    # DeepSeek-671B: reshape -> rsqrt -> multiply (RMSNorm)
    # %136 = reshape : (1x32xf32) -> (1x32x1xf32)
    # %137 = rsqrt(%136)
    # broadcast -> multiply with (1x32x512)
    pytest.param(
        (1, 32),
        (1, 32, 1),
        "rsqrt",
        "multiply",
        (1, 32, 1),
        id=_id("deepseek671b", (1, 32), (1, 32, 1), "rsqrt_multiply"),
    ),
    # Qwen3-32B: reshape -> rsqrt -> multiply (RMSNorm)
    # %867 = reshape : (32x17xf32) -> (544x1xf32)
    # %868 = rsqrt(%867)
    # broadcast -> multiply with (544x5120)
    pytest.param(
        (32, 17),
        (544, 1),
        "rsqrt",
        "multiply",
        (544, 1),
        id=_id("qwen3_32b", (32, 17), (544, 1), "rsqrt_multiply"),
    ),
    # GPT-OSS-120B: reshape -> cos -> multiply (RoPE)
    # %596 = reshape : (1x128x32xf32) -> (1x1x128x32xf32)
    # %597 = cos(%596)
    # %598 = multiply(%597, %5)
    pytest.param(
        (1, 128, 32),
        (1, 1, 128, 32),
        "cos",
        "multiply",
        (1, 1, 128, 32),
        id=_id("gpt_oss120b", (1, 128, 32), (1, 1, 128, 32), "cos_multiply"),
    ),
    # GPT-OSS-120B: reshape -> sin -> multiply (RoPE)
    pytest.param(
        (1, 128, 32),
        (1, 1, 128, 32),
        "sin",
        "multiply",
        (1, 1, 128, 32),
        id=_id("gpt_oss120b", (1, 128, 32), (1, 1, 128, 32), "sin_multiply"),
    ),
    # Qwen3-32B: reshape -> cos -> multiply (RoPE)
    pytest.param(
        (1, 17, 128),
        (1, 1, 17, 128),
        "cos",
        "multiply",
        (1, 1, 17, 128),
        id=_id("qwen3_32b", (1, 17, 128), (1, 1, 17, 128), "cos_multiply"),
    ),
]


@pytest.mark.parametrize(
    "input_shape, reshape_shape, unary_op, binary_op, rhs_shape",
    RESHAPE_UNARY_BINARY_CASES,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reshape_unary_then_binary(
    input_shape: Shape,
    reshape_shape: Tuple[int, ...],
    unary_op: str,
    binary_op: str,
    rhs_shape: Tuple[int, ...],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Reshape -> unary op -> binary op (e.g. RMSNorm / RoPE tail)."""

    def module(builder: TTIRBuilder):
        @builder.func([input_shape, rhs_shape], [dtype, dtype])
        def reshape_unary_binary(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            reshaped = builder.reshape(in0, list(reshape_shape))
            unary_fn = getattr(builder, unary_op)
            intermediate = unary_fn(reshaped)
            binary_fn = getattr(builder, binary_op)
            return binary_fn(intermediate, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


# ---------- binary -> reshape (compute then reshape) ----------

BINARY_THEN_RESHAPE_CASES: List = [
    # GPT-OSS-120B: add -> reshape (attention output reshape)
    # %589 = add(%583, %588) : tensor<128x1024xbf16>
    # %590 = reshape(%589) : (128x1024) -> (1x128x16x64)
    pytest.param(
        (128, 128),
        (128, 128),
        "add",
        (1, 128, 2, 64),
        id=_id("gpt_oss120b", (128, 128), (1, 128, 2, 64), "add_reshape"),
    ),
    # GLM-358B: add -> reshape (attention output reshape)
    # %41 = add(%35, %40) : tensor<32x1024xbf16>
    # %42 = reshape(%41) : (32x1024) -> (1x32x8x128)
    pytest.param(
        (32, 1024),
        (32, 1024),
        "add",
        (1, 32, 8, 128),
        id=_id("glm358b", (32, 1024), (1, 32, 8, 128), "add_reshape"),
    ),
    # GLM-358B: add -> reshape (QKV projection bias add + reshape)
    # %98 = add(%94, %97) : tensor<32x12288xbf16>
    # %100 = reshape(%99) : (32x12288) -> (1x32x96x128)  (after typecast)
    pytest.param(
        (32, 3072),
        (32, 3072),
        "add",
        (1, 32, 96, 32),
        id=_id("glm358b", (32, 3072), (1, 32, 96, 32), "add_reshape"),
    ),
    # Qwen3-32B: multiply -> reshape (RMSNorm output reshape)
    # %870 = multiply(%861, %869) : tensor<544x5120xf32>
    # reshape back to (32x17x...) for downstream
    pytest.param(
        (544, 64),
        (544, 64),
        "multiply",
        (32, 17, 64),
        id=_id("qwen3_32b", (544, 64), (32, 17, 64), "multiply_reshape"),
    ),
]


@pytest.mark.parametrize(
    "lhs_shape, rhs_shape, op_name, reshape_shape", BINARY_THEN_RESHAPE_CASES
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_binary_then_reshape(
    lhs_shape: Shape,
    rhs_shape: Tuple[int, ...],
    op_name: str,
    reshape_shape: Tuple[int, ...],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Binary op followed by reshape (e.g. attention output restructuring)."""

    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [dtype, dtype])
        def binary_reshape(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            op_fn = getattr(builder, op_name)
            result = op_fn(in0, in1)
            return builder.reshape(result, list(reshape_shape))

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


# ---------- reshape -> binary -> reshape (sandwich patterns) ----------

RESHAPE_BINARY_RESHAPE_CASES: List = [
    # GPT-OSS-120B: reshape 4D->2D, add, reshape 2D->4D (full attention bias)
    # %588 = reshape : (1x128x2x64) -> (128x128)
    # %589 = add
    # %590 = reshape : (128x128) -> (1x128x2x64)
    pytest.param(
        (1, 128, 2, 64),
        (128, 128),
        "add",
        (128, 128),
        (1, 128, 2, 64),
        id="gpt_oss120b__reshape_add_reshape__attn",
    ),
    # GLM-358B: reshape 4D->2D, add, reshape 2D->4D
    # %40 = reshape : (1x32x8x128) -> (32x1024)
    # %41 = add
    # %42 = reshape : (32x1024) -> (1x32x8x128)
    pytest.param(
        (1, 32, 8, 128),
        (32, 1024),
        "add",
        (32, 1024),
        (1, 32, 8, 128),
        id="glm358b__reshape_add_reshape__attn",
    ),
    # Qwen3-32B: reshape 4D->2D (flatten batch*seq), multiply, reshape back
    # Used in RMSNorm: flatten -> norm -> unflatten
    pytest.param(
        (32, 17, 1, 64),
        (544, 64),
        "multiply",
        (544, 64),
        (32, 17, 1, 64),
        id="qwen3_32b__reshape_multiply_reshape__rmsnorm",
    ),
]


@pytest.mark.parametrize(
    "input_shape, first_reshape, op_name, rhs_shape, final_reshape",
    RESHAPE_BINARY_RESHAPE_CASES,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reshape_binary_reshape(
    input_shape: Shape,
    first_reshape: Tuple[int, ...],
    op_name: str,
    rhs_shape: Tuple[int, ...],
    final_reshape: Tuple[int, ...],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Reshape -> binary op -> reshape (sandwich, e.g. attention bias path)."""

    def module(builder: TTIRBuilder):
        @builder.func([input_shape, rhs_shape], [dtype, dtype])
        def reshape_binary_reshape(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            reshaped_in = builder.reshape(in0, list(first_reshape))
            op_fn = getattr(builder, op_name)
            result = op_fn(reshaped_in, in1)
            return builder.reshape(result, list(final_reshape))

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


# ---------- dual-reshape + binary (both inputs reshaped) ----------

DUAL_RESHAPE_BINARY_CASES: List = [
    # DeepSeek-671B / GPT-OSS-120B style: both sides reshaped before multiply
    # E.g. RoPE: one input reshaped for cos, another for the value tensor
    pytest.param(
        (1, 32, 64),
        (1, 32, 1, 64),
        (32, 64),
        (1, 32, 1, 64),
        "multiply",
        id="deepseek671b__dual_reshape_multiply__rope",
    ),
    # Kimik2-1T: reshape scalar-like inputs, then outer-product-style multiply
    # %24 = reshape(%23) : (32xbf16) -> (32x1xbf16)
    # %26 = reshape(%arg0) : (32xbf16) -> (1x32xbf16)
    # %28 = multiply(%25, %27) -- after broadcasts
    pytest.param(
        (32,),
        (32, 1),
        (32,),
        (1, 32),
        "multiply",
        id="kimik2_1t__dual_reshape_multiply__outer",
    ),
]


@pytest.mark.parametrize(
    "lhs_input, lhs_reshape, rhs_input, rhs_reshape, op_name",
    DUAL_RESHAPE_BINARY_CASES,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dual_reshape_binary(
    lhs_input: Shape,
    lhs_reshape: Tuple[int, ...],
    rhs_input: Shape,
    rhs_reshape: Tuple[int, ...],
    op_name: str,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Both inputs are reshaped before a binary op."""

    def module(builder: TTIRBuilder):
        @builder.func([lhs_input, rhs_input], [dtype, dtype])
        def dual_reshape_binary(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            lhs = builder.reshape(in0, list(lhs_reshape))
            rhs = builder.reshape(in1, list(rhs_reshape))
            op_fn = getattr(builder, op_name)
            return op_fn(lhs, rhs)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


# ---------- multi-op composite patterns ----------


@pytest.mark.parametrize("target", ["ttmetal"])
def test_rmsnorm_pattern_glm358b(target: str, request, device):
    """Full RMSNorm pattern from GLM-358B.

    reshape(1x32x8 -> 1x32x8x1) -> rsqrt -> multiply with (1x32x8x128)
    Simulates the tail of RMSNorm: normalize -> scale.
    """
    input_shape = (1, 32, 8)
    weight_shape = (1, 32, 8, 1)
    dtype = torch.float32

    def module(builder: TTIRBuilder):
        @builder.func([input_shape, weight_shape], [dtype, dtype])
        def rmsnorm_tail(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            reshaped = builder.reshape(in0, [1, 32, 8, 1])
            normed = builder.rsqrt(reshaped)
            return builder.multiply(normed, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_rmsnorm_pattern_qwen3_32b(target: str, request, device):
    """Full RMSNorm pattern from Qwen3-32B.

    reshape(32x17 -> 544x1) -> rsqrt -> multiply with (544x64)
    Flattens batch*seq, normalizes, multiplies by weight.
    """
    input_shape = (32, 17)
    weight_shape = (544, 64)
    dtype = torch.float32

    def module(builder: TTIRBuilder):
        @builder.func([input_shape, weight_shape], [dtype, dtype])
        def rmsnorm_tail(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            reshaped = builder.reshape(in0, [544, 1])
            normed = builder.rsqrt(reshaped)
            # In the real model this broadcasts to 544x5120, use 544x64 proxy
            return builder.multiply(normed, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_rope_cos_sin_pattern_gpt_oss120b(target: str, request, device):
    """Rotary embedding cos/sin pattern from GPT-OSS-120B.

    reshape(1x128x32 -> 1x1x128x32) -> cos -> multiply
    reshape(1x128x32 -> 1x1x128x32) -> sin -> multiply
    subtract(cos_result, sin_result)
    """
    pos_shape = (1, 128, 32)
    scale_shape = (1, 1, 128, 32)
    dtype = torch.float32

    def module(builder: TTIRBuilder):
        @builder.func([pos_shape, scale_shape, scale_shape], [dtype, dtype, dtype])
        def rope_cos_sin(
            positions: Operand,
            cos_scale: Operand,
            sin_scale: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            reshaped = builder.reshape(positions, [1, 1, 128, 32])
            cos_val = builder.cos(reshaped)
            sin_val = builder.sin(reshaped)
            cos_out = builder.multiply(cos_val, cos_scale)
            sin_out = builder.multiply(sin_val, sin_scale)
            return builder.subtract(cos_out, sin_out)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_attn_bias_add_reshape_pattern_glm358b(target: str, request, device):
    """Attention bias add + reshape pattern from GLM-358B.

    Two paths reshaped to 2D, added, then reshaped back to 4D.
    reshape(1x32x8x128 -> 32x1024) + matmul_out(32x1024) -> add -> reshape(32x1024 -> 1x32x8x128)
    """
    proj_shape = (32, 1024)
    bias_input_shape = (1, 32, 8, 128)
    dtype = torch.float32

    def module(builder: TTIRBuilder):
        @builder.func([proj_shape, bias_input_shape], [dtype, dtype])
        def attn_bias_add_reshape(
            proj: Operand,
            bias_4d: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            bias_2d = builder.reshape(bias_4d, [32, 1024])
            added = builder.add(proj, bias_2d)
            return builder.reshape(added, [1, 32, 8, 128])

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )
