# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import math
from typing import List, Optional
from dataclasses import dataclass

import pytest
import torch

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import clear_device_cache, get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


# ---------------------------------------------------------------------------
# Shape definitions
# ---------------------------------------------------------------------------


@dataclass
class SDPAShapes:
    batch: int
    q_heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    head_dim: int

    @property
    def query(self):
        return (self.batch, self.q_heads, self.q_seq, self.head_dim)

    @property
    def key(self):
        return (self.batch, self.kv_heads, self.kv_seq, self.head_dim)

    @property
    def value(self):
        return (self.batch, self.kv_heads, self.kv_seq, self.head_dim)

    @property
    def mask(self):
        return (self.batch, 1, self.q_seq, self.kv_seq)

    @property
    def is_decode(self):
        return self.q_seq == 1

    @property
    def is_gqa(self):
        return self.q_heads != self.kv_heads

    @property
    def scale(self):
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def split_scale(self):
        return 1.0 / math.sqrt(math.sqrt(self.head_dim))


DECODE_SHAPES = [
    pytest.param(
        SDPAShapes(batch=32, q_heads=32, kv_heads=8, q_seq=1, kv_seq=128, head_dim=64),
        id="gqa_decode",
    ),
    pytest.param(
        SDPAShapes(batch=32, q_heads=32, kv_heads=32, q_seq=1, kv_seq=128, head_dim=64),
        id="mha_decode",
    ),
    pytest.param(
        SDPAShapes(batch=8, q_heads=32, kv_heads=1, q_seq=1, kv_seq=128, head_dim=64),
        id="mqa_decode",
    ),
    pytest.param(
        SDPAShapes(batch=8, q_heads=32, kv_heads=8, q_seq=1, kv_seq=256, head_dim=128),
        id="gqa_decode_large_hd",
    ),
    pytest.param(
        SDPAShapes(batch=32, q_heads=64, kv_heads=8, q_seq=1, kv_seq=128, head_dim=64),
        id="high_gqa_decode",
    ),
]

PREFILL_SHAPES = [
    pytest.param(
        SDPAShapes(batch=1, q_heads=32, kv_heads=8, q_seq=128, kv_seq=128, head_dim=64),
        id="gqa_prefill",
    ),
    pytest.param(
        SDPAShapes(
            batch=1, q_heads=12, kv_heads=12, q_seq=128, kv_seq=128, head_dim=64
        ),
        id="mha_prefill",
    ),
]

ALL_SHAPES = DECODE_SHAPES + PREFILL_SHAPES


# ---------------------------------------------------------------------------
# Composable SDPA building blocks
# ---------------------------------------------------------------------------


def gqa_broadcast_kv(key, value, q_heads, builder, unit_attrs=None):
    """Expand KV heads to match Q heads via reshape -> broadcast -> reshape."""
    k_shape = builder.get_shape(key)
    v_shape = builder.get_shape(value)
    batch, kv_heads, kv_seq, head_dim = k_shape

    if q_heads == kv_heads:
        return key, value

    assert q_heads % kv_heads == 0
    num_repeats = q_heads // kv_heads

    # Expand K
    k_5d = builder.reshape(
        key, [batch, kv_heads, 1, kv_seq, head_dim], unit_attrs=unit_attrs
    )
    k_broadcast = builder.broadcast(
        k_5d, [1, 1, num_repeats, 1, 1], unit_attrs=unit_attrs
    )
    key = builder.reshape(
        k_broadcast, [batch, q_heads, kv_seq, head_dim], unit_attrs=unit_attrs
    )

    # Expand V
    v_5d = builder.reshape(
        value, [batch, kv_heads, 1, kv_seq, head_dim], unit_attrs=unit_attrs
    )
    v_broadcast = builder.broadcast(
        v_5d, [1, 1, num_repeats, 1, 1], unit_attrs=unit_attrs
    )
    value = builder.reshape(
        v_broadcast, [batch, q_heads, kv_seq, head_dim], unit_attrs=unit_attrs
    )

    return key, value


def pre_scale(tensor, scale, builder, dtype=torch.float32, unit_attrs=None):
    """Typecast to dtype and multiply by scalar constant."""
    tensor_f = builder.typecast(tensor, dtype, unit_attrs=unit_attrs)
    shape = builder.get_shape(tensor_f)
    scale_shape = [1] * len(shape)
    scale_data = torch.full(scale_shape, scale, dtype=dtype)
    scale_tensor = builder.constant(scale_data, unit_attrs=unit_attrs)
    return builder.multiply(tensor_f, scale_tensor, unit_attrs=unit_attrs)


def post_scale_scores(scores, scale, builder, unit_attrs=None):
    """Multiply QK^T scores by scalar (via full op)."""
    scores_shape = builder.get_shape(scores)
    scale_shape = [1] * len(scores_shape)
    scale_tensor = builder.full(
        scale_shape, torch.bfloat16, scale, unit_attrs=unit_attrs
    )
    return builder.multiply(scores, scale_tensor, unit_attrs=unit_attrs)


def compute_scores(query, key, builder, mask=None, unit_attrs=None):
    """Transpose K, compute Q @ K^T, optionally add mask."""
    key_t = builder.transpose(key, dim0=-2, dim1=-1)
    scores = builder.matmul(query, key_t, unit_attrs=unit_attrs)
    if mask is not None:
        scores = builder.add(scores, mask, unit_attrs=unit_attrs)
    return scores


def simple_softmax(
    scores, builder, in_dtype=torch.float32, out_dtype=torch.bfloat16, unit_attrs=None
):
    """typecast -> max -> sub -> exp -> sum -> div -> typecast"""
    scores_f = builder.typecast(scores, in_dtype, unit_attrs=unit_attrs)
    scores_shape = builder.get_shape(scores_f)
    last_dim = scores_shape[-1]

    # max along last dim
    s_max = builder.max(scores_f, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs)
    s_max_shape = builder.get_shape(s_max)
    s_max_4d = builder.reshape(s_max, list(s_max_shape) + [1], unit_attrs=unit_attrs)
    s_max_bc = builder.broadcast(s_max_4d, [1, 1, 1, last_dim], unit_attrs=unit_attrs)

    shifted = builder.subtract(scores_f, s_max_bc, unit_attrs=unit_attrs)
    exp_vals = builder.exp(shifted, unit_attrs=unit_attrs)

    # sum along last dim
    s_sum = builder.sum(exp_vals, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs)
    s_sum_shape = builder.get_shape(s_sum)
    s_sum_4d = builder.reshape(s_sum, list(s_sum_shape) + [1], unit_attrs=unit_attrs)
    s_sum_bc = builder.broadcast(s_sum_4d, [1, 1, 1, last_dim], unit_attrs=unit_attrs)

    softmax_out = builder.div(exp_vals, s_sum_bc, unit_attrs=unit_attrs)
    return builder.typecast(softmax_out, out_dtype, unit_attrs=unit_attrs)


def nan_safe_softmax(
    scores, builder, in_dtype=torch.float32, out_dtype=torch.bfloat16, unit_attrs=None
):
    """Simple softmax + eq/-inf -> logical_not -> reduce_or -> where(zeros) for masked rows."""
    scores_f = builder.typecast(scores, in_dtype, unit_attrs=unit_attrs)
    scores_shape = builder.get_shape(scores_f)
    last_dim = scores_shape[-1]

    # Detect all-masked rows (all -inf)
    neg_inf_data = torch.full(scores_shape, float("-inf"), dtype=in_dtype)
    neg_inf = builder.constant(neg_inf_data, unit_attrs=unit_attrs)
    is_neg_inf = builder.eq(scores_f, neg_inf, unit_attrs=unit_attrs)
    is_valid = builder.logical_not(is_neg_inf, unit_attrs=unit_attrs)
    any_valid = builder.reduce_or(
        is_valid, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs
    )
    any_valid_shape = builder.get_shape(any_valid)
    any_valid_4d = builder.reshape(
        any_valid, any_valid_shape + [1], unit_attrs=unit_attrs
    )
    all_masked = builder.logical_not(any_valid_4d, unit_attrs=unit_attrs)
    all_masked_bc = builder.broadcast(
        all_masked, [1, 1, 1, last_dim], unit_attrs=unit_attrs
    )

    # Standard softmax: max -> sub -> exp -> sum -> div
    s_max = builder.max(scores_f, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs)
    s_max_shape = builder.get_shape(s_max)
    s_max_4d = builder.reshape(s_max, list(s_max_shape) + [1], unit_attrs=unit_attrs)
    s_max_bc = builder.broadcast(s_max_4d, [1, 1, 1, last_dim], unit_attrs=unit_attrs)

    shifted = builder.subtract(scores_f, s_max_bc, unit_attrs=unit_attrs)
    exp_vals = builder.exp(shifted, unit_attrs=unit_attrs)

    s_sum = builder.sum(exp_vals, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs)
    s_sum_shape = builder.get_shape(s_sum)
    s_sum_4d = builder.reshape(s_sum, list(s_sum_shape) + [1], unit_attrs=unit_attrs)
    s_sum_bc = builder.broadcast(s_sum_4d, [1, 1, 1, last_dim], unit_attrs=unit_attrs)

    softmax_out = builder.div(exp_vals, s_sum_bc, unit_attrs=unit_attrs)

    # Replace all-masked rows with zeros
    zeros_data = torch.zeros(scores_shape, dtype=in_dtype)
    zeros = builder.constant(zeros_data, unit_attrs=unit_attrs)
    softmax_out = builder.where(
        all_masked_bc, zeros, softmax_out, unit_attrs=unit_attrs
    )

    return builder.typecast(softmax_out, out_dtype, unit_attrs=unit_attrs)


def attention_output(
    attn_weights,
    value,
    builder,
    in_dtype=None,
    out_dtype=torch.bfloat16,
    unit_attrs=None,
):
    """Matmul attn_weights @ V, optional typecast."""
    if in_dtype is not None:
        value = builder.typecast(value, in_dtype, unit_attrs=unit_attrs)
    output = builder.matmul(attn_weights, value, unit_attrs=unit_attrs)
    return builder.typecast(output, out_dtype, unit_attrs=unit_attrs)


# ---------------------------------------------------------------------------
# Golden computation
# ---------------------------------------------------------------------------


def build_torch_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    enable_gqa = q_heads != kv_heads
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=scale, enable_gqa=enable_gqa
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def check_op(mlir_file: str, op_name: str) -> bool:
    op_name = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if op_name in line:
                return True
    return False


def assert_sdpa_fused(artifact_dir: str, q_seq: int):
    output_path = os.path.join(artifact_dir, "ttnn_compiled.mlir")
    if q_seq == 1:
        assert check_op(output_path, "scaled_dot_product_attention_decode")
    else:
        assert check_op(output_path, "scaled_dot_product_attention")


def compile_and_run_sdpa(module_fn, target, request):
    return compile_and_execute_ttir(
        module_fn,
        target=target,
        **get_request_kwargs(request),
        device=lambda: (clear_device_cache(), request.getfixturevalue("device"))[1],
        pipeline_options=["enable-optimizer=true"],
        save_artifacts=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sdpa_shapes", ALL_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_split_scale_nan_safe(sdpa_shapes: SDPAShapes, target: str, request):
    """
    Qwen / Phi / BERT / ViT pattern.
    Pattern: (Q * scale) @ (K^T * scale) -> mask -> nan_safe_softmax -> @ V
    """
    input_shapes = [
        sdpa_shapes.query,
        sdpa_shapes.key,
        sdpa_shapes.value,
        sdpa_shapes.mask,
    ]
    dtypes = [torch.bfloat16] * 4

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(query, key, value, mask, builder: TTIRBuilder, unit_attrs=None):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            m_data = torch.zeros(sdpa_shapes.mask, dtype=torch.bfloat16)

            golden = build_torch_golden(
                q_data, k_data, v_data, scale=sdpa_shapes.scale, attention_mask=m_data
            )

            # Build TTIR: split pre-scale Q and K, GQA expand, compute scores, nan-safe softmax
            q_scaled = pre_scale(
                query, sdpa_shapes.split_scale, builder, unit_attrs=unit_attrs
            )
            key_exp, value_exp = gqa_broadcast_kv(
                key, value, sdpa_shapes.q_heads, builder, unit_attrs=unit_attrs
            )
            k_scaled = pre_scale(
                key_exp,
                sdpa_shapes.split_scale,
                builder,
                dtype=torch.float32,
                unit_attrs=unit_attrs,
            )
            k_t = builder.transpose(k_scaled, dim0=-2, dim1=-1)
            scores = builder.matmul(q_scaled, k_t, unit_attrs=unit_attrs)
            mask_f32 = builder.typecast(mask, torch.float32, unit_attrs=unit_attrs)
            scores = builder.add(scores, mask_f32, unit_attrs=unit_attrs)
            attn = nan_safe_softmax(
                scores,
                builder,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                unit_attrs=unit_attrs,
            )
            v_f32 = builder.typecast(value_exp, torch.float32, unit_attrs=unit_attrs)
            output = builder.matmul(attn, v_f32, unit_attrs=unit_attrs)
            result = builder.typecast(output, torch.bfloat16, unit_attrs=unit_attrs)

            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data, mask: m_data},
                {result: golden},
            )
            return result

    artifact_dir = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(artifact_dir, sdpa_shapes.q_seq)


@pytest.mark.parametrize("sdpa_shapes", ALL_SHAPES)
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_post_scale_simple(
    sdpa_shapes: SDPAShapes, use_mask: bool, target: str, request
):
    """
    SegFormer-like / simple pattern.
    Pattern: Q @ K^T -> scale -> [mask] -> simple_softmax -> @ V
    """
    input_shapes = [sdpa_shapes.query, sdpa_shapes.key, sdpa_shapes.value]
    if use_mask:
        input_shapes.append(sdpa_shapes.mask)
    dtypes = [torch.bfloat16] * len(input_shapes)

    def module_no_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(query, key, value, builder: TTIRBuilder, unit_attrs=None):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)

            golden = build_torch_golden(q_data, k_data, v_data, scale=sdpa_shapes.scale)

            key_exp, value_exp = gqa_broadcast_kv(
                key, value, sdpa_shapes.q_heads, builder, unit_attrs=unit_attrs
            )
            scores = compute_scores(query, key_exp, builder, unit_attrs=unit_attrs)
            scores = post_scale_scores(
                scores, sdpa_shapes.scale, builder, unit_attrs=unit_attrs
            )
            attn = simple_softmax(scores, builder, unit_attrs=unit_attrs)
            result = builder.matmul(attn, value_exp, unit_attrs=unit_attrs)

            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data},
                {result: golden},
            )
            return result

    def module_with_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(query, key, value, mask, builder: TTIRBuilder, unit_attrs=None):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            m_data = torch.zeros(sdpa_shapes.mask, dtype=torch.bfloat16)

            golden = build_torch_golden(
                q_data, k_data, v_data, scale=sdpa_shapes.scale, attention_mask=m_data
            )

            key_exp, value_exp = gqa_broadcast_kv(
                key, value, sdpa_shapes.q_heads, builder, unit_attrs=unit_attrs
            )
            scores = compute_scores(query, key_exp, builder, unit_attrs=unit_attrs)
            scores = post_scale_scores(
                scores, sdpa_shapes.scale, builder, unit_attrs=unit_attrs
            )
            scores = builder.add(scores, mask, unit_attrs=unit_attrs)
            attn = simple_softmax(scores, builder, unit_attrs=unit_attrs)
            result = builder.matmul(attn, value_exp, unit_attrs=unit_attrs)

            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data, mask: m_data},
                {result: golden},
            )
            return result

    artifact_dir = compile_and_run_sdpa(
        module_with_mask if use_mask else module_no_mask, target, request
    )
    assert_sdpa_fused(artifact_dir, sdpa_shapes.q_seq)


@pytest.mark.parametrize("sdpa_shapes", ALL_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_pre_scale_q_nan_safe(sdpa_shapes: SDPAShapes, target: str, request):
    """
    LLaMA decode pattern.
    Pattern: (Q * scale) @ K^T -> mask -> nan_safe_softmax -> @ V
    """
    input_shapes = [
        sdpa_shapes.query,
        sdpa_shapes.key,
        sdpa_shapes.value,
        sdpa_shapes.mask,
    ]
    dtypes = [torch.bfloat16] * 4

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(query, key, value, mask, builder: TTIRBuilder, unit_attrs=None):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            m_data = torch.zeros(sdpa_shapes.mask, dtype=torch.bfloat16)

            golden = build_torch_golden(
                q_data, k_data, v_data, scale=sdpa_shapes.scale, attention_mask=m_data
            )

            # Pre-scale Q only, then GQA expand K/V
            q_scaled = pre_scale(
                query, sdpa_shapes.scale, builder, unit_attrs=unit_attrs
            )
            key_exp, value_exp = gqa_broadcast_kv(
                key, value, sdpa_shapes.q_heads, builder, unit_attrs=unit_attrs
            )
            k_f32 = builder.typecast(key_exp, torch.float32, unit_attrs=unit_attrs)
            k_t = builder.transpose(k_f32, dim0=-2, dim1=-1)
            scores = builder.matmul(q_scaled, k_t, unit_attrs=unit_attrs)
            mask_f32 = builder.typecast(mask, torch.float32, unit_attrs=unit_attrs)
            scores = builder.add(scores, mask_f32, unit_attrs=unit_attrs)
            attn = nan_safe_softmax(
                scores,
                builder,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                unit_attrs=unit_attrs,
            )
            v_f32 = builder.typecast(value_exp, torch.float32, unit_attrs=unit_attrs)
            output = builder.matmul(attn, v_f32, unit_attrs=unit_attrs)
            result = builder.typecast(output, torch.bfloat16, unit_attrs=unit_attrs)

            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data, mask: m_data},
                {result: golden},
            )
            return result

    artifact_dir = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(artifact_dir, sdpa_shapes.q_seq)


@pytest.mark.parametrize("sdpa_shapes", ALL_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_pre_scale_k_nan_safe(sdpa_shapes: SDPAShapes, target: str, request):
    """
    LLaMA prefill / Falcon / Gemma / Mistral pattern.
    Pattern: Q @ (K^T * scale) -> mask -> nan_safe_softmax -> @ V
    """
    input_shapes = [
        sdpa_shapes.query,
        sdpa_shapes.key,
        sdpa_shapes.value,
        sdpa_shapes.mask,
    ]
    dtypes = [torch.bfloat16] * 4

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(query, key, value, mask, builder: TTIRBuilder, unit_attrs=None):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            m_data = torch.zeros(sdpa_shapes.mask, dtype=torch.bfloat16)

            golden = build_torch_golden(
                q_data, k_data, v_data, scale=sdpa_shapes.scale, attention_mask=m_data
            )

            # GQA expand, then pre-scale K only
            key_exp, value_exp = gqa_broadcast_kv(
                key, value, sdpa_shapes.q_heads, builder, unit_attrs=unit_attrs
            )
            k_scaled = pre_scale(
                key_exp, sdpa_shapes.scale, builder, unit_attrs=unit_attrs
            )
            k_t = builder.transpose(k_scaled, dim0=-2, dim1=-1)
            q_f32 = builder.typecast(query, torch.float32, unit_attrs=unit_attrs)
            scores = builder.matmul(q_f32, k_t, unit_attrs=unit_attrs)
            mask_f32 = builder.typecast(mask, torch.float32, unit_attrs=unit_attrs)
            scores = builder.add(scores, mask_f32, unit_attrs=unit_attrs)
            attn = nan_safe_softmax(
                scores,
                builder,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                unit_attrs=unit_attrs,
            )
            v_f32 = builder.typecast(value_exp, torch.float32, unit_attrs=unit_attrs)
            output = builder.matmul(attn, v_f32, unit_attrs=unit_attrs)
            result = builder.typecast(output, torch.bfloat16, unit_attrs=unit_attrs)

            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data, mask: m_data},
                {result: golden},
            )
            return result

    artifact_dir = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(artifact_dir, sdpa_shapes.q_seq)


@pytest.mark.parametrize("sdpa_shapes", ALL_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_simple_softmax(sdpa_shapes: SDPAShapes, target: str, request):
    """
    Swin-like pattern: simple softmax, no NaN handling.
    Pattern: Q @ K^T -> scale -> mask -> simple_softmax -> @ V
    """
    input_shapes = [
        sdpa_shapes.query,
        sdpa_shapes.key,
        sdpa_shapes.value,
        sdpa_shapes.mask,
    ]
    dtypes = [torch.bfloat16] * 4

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(query, key, value, mask, builder: TTIRBuilder, unit_attrs=None):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            m_data = torch.zeros(sdpa_shapes.mask, dtype=torch.bfloat16)

            golden = build_torch_golden(
                q_data, k_data, v_data, scale=sdpa_shapes.scale, attention_mask=m_data
            )

            key_exp, value_exp = gqa_broadcast_kv(
                key, value, sdpa_shapes.q_heads, builder, unit_attrs=unit_attrs
            )
            scores = compute_scores(query, key_exp, builder, unit_attrs=unit_attrs)
            scores = post_scale_scores(
                scores, sdpa_shapes.scale, builder, unit_attrs=unit_attrs
            )
            scores = builder.add(scores, mask, unit_attrs=unit_attrs)
            attn = simple_softmax(scores, builder, unit_attrs=unit_attrs)
            result = builder.matmul(attn, value_exp, unit_attrs=unit_attrs)

            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data, mask: m_data},
                {result: golden},
            )
            return result

    artifact_dir = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(artifact_dir, sdpa_shapes.q_seq)
