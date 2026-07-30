# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import math
from typing import List, Optional
from dataclasses import dataclass

import pytest
import torch

from builder.base.builder_utils import Operand, Shape, DeferredDevice
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = [
    pytest.mark.frontend("ttir"),
    pytest.mark.skip_config(
        ("p150",),
        ("p300",),
        reason="Optimizer mock device grid mismatch on Blackhole (https://github.com/tenstorrent/tt-mlir/issues/7809)",
    ),
]


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

# Real model shapes extracted from tt-xla CI TTIR artifacts (benchmark run),
# tagged by model + phase (mirrors test_rope_fusing.py). These bring head_dims
# and GQA ratios the synthetic set above doesn't cover (hd256, 3:1 / 2:1 / 4:1
# GQA) plus a cross-seq prefill chunk (Sq < Sk). Seqs are kept >= 32 to avoid
# the sub-tile SDPA re-query explosion (see project_sdpa_subtile_hang).
REAL_MODEL_SHAPES = [
    # llama_3_2_1b prefill chunk: GQA 32/8, hd64, cross-seq (Sq < Sk with KV cache)
    pytest.param(
        SDPAShapes(batch=1, q_heads=32, kv_heads=8, q_seq=32, kv_seq=128, head_dim=64),
        id="llama_3_2_1b-prefill-crossseq",
    ),
    # mistral_7b prefill: GQA 32/8 (4:1), hd128
    pytest.param(
        SDPAShapes(
            batch=1, q_heads=32, kv_heads=8, q_seq=128, kv_seq=128, head_dim=128
        ),
        id="mistral_7b-prefill",
    ),
    # falcon3_7b prefill: GQA 12/4 (3:1), hd256
    pytest.param(
        SDPAShapes(
            batch=1, q_heads=12, kv_heads=4, q_seq=128, kv_seq=128, head_dim=256
        ),
        id="falcon3_7b-prefill",
    ),
    # falcon3_7b decode: GQA 12/4 (3:1), hd256
    pytest.param(
        SDPAShapes(batch=32, q_heads=12, kv_heads=4, q_seq=1, kv_seq=128, head_dim=256),
        id="falcon3_7b-decode",
    ),
    # gemma2_2b prefill: GQA 8/4 (2:1), hd256
    pytest.param(
        SDPAShapes(batch=1, q_heads=8, kv_heads=4, q_seq=128, kv_seq=128, head_dim=256),
        id="gemma2_2b-prefill",
    ),
    # phi_1 prefill: MHA 32/32, hd64
    pytest.param(
        SDPAShapes(
            batch=1, q_heads=32, kv_heads=32, q_seq=128, kv_seq=128, head_dim=64
        ),
        id="phi_1-prefill",
    ),
]

ALL_SHAPES = DECODE_SHAPES + PREFILL_SHAPES + REAL_MODEL_SHAPES

# Only GQA shapes (Hq != Hkv) — used by the repeat_interleave test, whose whole
# point is exercising the native-GQA peel (a no-op when Hq == Hkv).
GQA_SHAPES = [p for p in ALL_SHAPES if p.values[0].is_gqa]

# Attention-sink shapes are MHA (Hq == Hkv) so the per-head sink is [1, Hq, 1, 1]
# with no GQA expansion tangled in. gpt_oss-like: hd64, small head count.
SINK_SHAPES = [
    pytest.param(
        SDPAShapes(batch=1, q_heads=8, kv_heads=8, q_seq=128, kv_seq=128, head_dim=64),
        id="gpt_oss-prefill-sink",
    ),
    pytest.param(
        SDPAShapes(batch=32, q_heads=8, kv_heads=8, q_seq=1, kv_seq=128, head_dim=64),
        id="gpt_oss-decode-sink",
    ),
]


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


def compute_scores_permute(query, key, builder, mask=None, unit_attrs=None):
    """Kᵀ via ttir.permute (dot_general form) instead of ttir.transpose.

    Real models lower Q·Kᵀ through dot_general, which decomposes Kᵀ to a
    ttir.permute swapping the last two dims — not a ttir.transpose. The matcher
    must recognize this form (isLastTwoDimsPermute) or it no-ops on real models.
    """
    key_t = builder.permute(key, [0, 1, 3, 2], unit_attrs=unit_attrs)
    scores = builder.matmul(query, key_t, unit_attrs=unit_attrs)
    if mask is not None:
        scores = builder.add(scores, mask, unit_attrs=unit_attrs)
    return scores


def gqa_repeat_interleave_kv(key, value, q_heads, builder, unit_attrs=None):
    """Expand KV heads to Q heads via ttir.repeat_interleave on the heads dim.

    Unlike gqa_broadcast_kv (reshape→broadcast→reshape, which fuses as MHA),
    this is the form the matcher's detectGqaExpansion peels to feed the native
    Hkv-head tensors, so the op stays GQA-native.
    """
    k_shape = builder.get_shape(key)
    kv_heads = k_shape[1]
    if q_heads == kv_heads:
        return key, value

    assert q_heads % kv_heads == 0
    num_repeats = q_heads // kv_heads

    key = builder.repeat_interleave(key, num_repeats, 1, unit_attrs=unit_attrs)
    value = builder.repeat_interleave(value, num_repeats, 1, unit_attrs=unit_attrs)
    return key, value


def div_scale_scores(scores, divisor, builder, unit_attrs=None):
    """Scale QK^T scores by dividing by a scalar (via full op) -> scale = 1/divisor.

    Exercises the matcher's DivOp scale-peel path (scale := 1/divisor).
    """
    scores_shape = builder.get_shape(scores)
    div_shape = [1] * len(scores_shape)
    div_tensor = builder.full(div_shape, torch.bfloat16, divisor, unit_attrs=unit_attrs)
    return builder.div(scores, div_tensor, unit_attrs=unit_attrs)


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


def build_sink_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Faithful attention-sink reference (MHA).

    The sink is an extra per-head logit column appended to the scores; it
    participates in the softmax denominator but is dropped before the value
    matmul. Kept faithful (sink is a raw post-scale logit) — the kernel's 1/scale
    sink compensation is applied in TTIRToTTNN lowering, not here
    (project_sdpa_sink_scale_compensation).
    """
    scores = (query.float() @ key.float().transpose(-2, -1)) * scale
    scores = scores + attention_mask.float()
    batch, heads, q_seq, kv_seq = scores.shape
    sink_col = sink.float().expand(batch, heads, q_seq, 1)
    augmented = torch.cat([scores, sink_col], dim=-1)
    weights = torch.softmax(augmented, dim=-1)[..., :kv_seq]
    return (weights @ value.float()).to(query.dtype)


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


def assert_sdpa_fused(mlir_path: str, q_seq: int):
    if q_seq == 1:
        assert check_op(mlir_path, "scaled_dot_product_attention_decode")
    else:
        assert check_op(mlir_path, "scaled_dot_product_attention")


def compile_and_run_sdpa(module_fn, target, request):
    return compile_and_execute_ttir(
        module_fn,
        target=target,
        **get_request_kwargs(request),
        device=DeferredDevice(request),
        pipeline_options=["optimization-level=1"],
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

    mlir_path = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(mlir_path, sdpa_shapes.q_seq)


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

    mlir_path = compile_and_run_sdpa(
        module_with_mask if use_mask else module_no_mask, target, request
    )
    assert_sdpa_fused(mlir_path, sdpa_shapes.q_seq)


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

    mlir_path = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(mlir_path, sdpa_shapes.q_seq)


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

    mlir_path = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(mlir_path, sdpa_shapes.q_seq)


@pytest.mark.parametrize("sdpa_shapes", ALL_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_kt_via_permute(sdpa_shapes: SDPAShapes, target: str, request):
    """
    dot_general lowering form (LLaMA / Mistral / Qwen).
    Kᵀ is a ttir.permute (last-two-dims swap), not a ttir.transpose — the form
    real models lower to. The matcher no-ops on real models without matching it.
    Pattern: Q @ permute(K) -> scale -> mask -> simple_softmax -> @ V
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
            scores = compute_scores_permute(
                query, key_exp, builder, unit_attrs=unit_attrs
            )
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

    mlir_path = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(mlir_path, sdpa_shapes.q_seq)


@pytest.mark.parametrize("sdpa_shapes", GQA_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_gqa_repeat_interleave_div_scale(
    sdpa_shapes: SDPAShapes, target: str, request
):
    """
    Native-GQA form. K/V are expanded via ttir.repeat_interleave (the form the
    matcher peels to feed the native Hkv-head tensors, unlike the broadcast
    expansion), and the scale is applied as a division scores / sqrt(d) rather
    than a multiply (matcher's DivOp scale-peel).
    Pattern: Q @ K^T -> / sqrt(d) -> mask -> simple_softmax -> @ V
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

            key_exp, value_exp = gqa_repeat_interleave_kv(
                key, value, sdpa_shapes.q_heads, builder, unit_attrs=unit_attrs
            )
            scores = compute_scores(query, key_exp, builder, unit_attrs=unit_attrs)
            scores = div_scale_scores(
                scores, math.sqrt(sdpa_shapes.head_dim), builder, unit_attrs=unit_attrs
            )
            scores = builder.add(scores, mask, unit_attrs=unit_attrs)
            attn = simple_softmax(scores, builder, unit_attrs=unit_attrs)
            result = builder.matmul(attn, value_exp, unit_attrs=unit_attrs)

            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data, mask: m_data},
                {result: golden},
            )
            return result

    mlir_path = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(mlir_path, sdpa_shapes.q_seq)


@pytest.mark.parametrize("sdpa_shapes", SINK_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_attention_sink(sdpa_shapes: SDPAShapes, target: str, request):
    """
    Attention sink (gpt_oss). A per-head sink logit [1, Hq, 1, 1] is concat'd as
    an extra score column before softmax, then sliced off after — folded into
    the op's attention_sink operand. The sink participates in the softmax
    denominator but not the value matmul.
    Pattern: concat(Q@K^T*scale + mask, sink) -> softmax -> slice[..., :Sk] -> @ V
    """
    sink_shape = (1, sdpa_shapes.q_heads, 1, 1)
    input_shapes = [
        sdpa_shapes.query,
        sdpa_shapes.key,
        sdpa_shapes.value,
        sdpa_shapes.mask,
        sink_shape,
    ]
    dtypes = [torch.bfloat16] * 5

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(query, key, value, mask, sink, builder: TTIRBuilder, unit_attrs=None):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            m_data = torch.zeros(sdpa_shapes.mask, dtype=torch.bfloat16)
            sink_data = torch.randn(sink_shape, dtype=torch.bfloat16)

            golden = build_sink_golden(
                q_data, k_data, v_data, sdpa_shapes.scale, m_data, sink_data
            )

            scores = compute_scores(query, key, builder, unit_attrs=unit_attrs)
            scores = post_scale_scores(
                scores, sdpa_shapes.scale, builder, unit_attrs=unit_attrs
            )
            scores = builder.add(scores, mask, unit_attrs=unit_attrs)

            # Sink logit as an extra column: broadcast [1, Hq, 1, 1] -> [B, Hq, Sq, 1]
            # then concat onto the kv (last) axis.
            sink_bc = builder.broadcast(
                sink,
                [sdpa_shapes.batch, 1, sdpa_shapes.q_seq, 1],
                unit_attrs=unit_attrs,
            )
            augmented = builder.concat([scores, sink_bc], dim=3, unit_attrs=unit_attrs)
            attn = simple_softmax(augmented, builder, unit_attrs=unit_attrs)
            # Drop the sink column: keep [..., :Sk].
            attn_sliced = builder.slice(
                attn,
                begins=[0, 0, 0, 0],
                ends=[
                    sdpa_shapes.batch,
                    sdpa_shapes.q_heads,
                    sdpa_shapes.q_seq,
                    sdpa_shapes.kv_seq,
                ],
                step=[1, 1, 1, 1],
                unit_attrs=unit_attrs,
            )
            result = builder.matmul(attn_sliced, value, unit_attrs=unit_attrs)

            builder.set_goldens(
                {
                    query: q_data,
                    key: k_data,
                    value: v_data,
                    mask: m_data,
                    sink: sink_data,
                },
                {result: golden},
            )
            return result

    mlir_path = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(mlir_path, sdpa_shapes.q_seq)


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

    mlir_path = compile_and_run_sdpa(module, target, request)
    assert_sdpa_fused(mlir_path, sdpa_shapes.q_seq)
