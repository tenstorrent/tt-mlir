# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
from typing import List, Optional
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str) -> bool:
    op_name = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if op_name in line:
                return True
    return False


def build_torch_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build golden output using PyTorch's scaled_dot_product_attention.
    Supports standard attention and Grouped-Query Attention (GQA).
    """
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    enable_gqa = q_heads != kv_heads

    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=scale, enable_gqa=enable_gqa
    )


def build_ttir(
    query: Operand,
    key: Operand,
    value: Operand,
    builder: TTIRBuilder,
    scale: Optional[float] = None,
    attention_mask: Optional[Operand] = None,
    unit_attrs: Optional[List[str]] = None,
):
    """
    Build TTIR representation of SDPA pattern:
    - Broadcast K/V heads if GQA (query has more heads than K/V)
    - Q @ K^T matmul
    - Scale (multiply)
    - Add mask (optional)
    - Softmax
    - @ V matmul
    """
    # Handle Grouped-Query Attention: broadcast K/V heads to match Q heads if needed
    q_shape = builder.get_shape(query)
    k_shape = builder.get_shape(key)
    q_heads = q_shape[1]
    kv_heads = k_shape[1]

    if q_heads != kv_heads:
        # GQA: repeat each K/V head to match Q heads
        assert (
            q_heads % kv_heads == 0
        ), f"Q heads ({q_heads}) must be divisible by KV heads ({kv_heads})"
        num_repeats = q_heads // kv_heads
        # Repeat K and V along the head dimension (dim=1)
        key = builder.repeat_interleave(
            key, repeats=num_repeats, dim=1, unit_attrs=unit_attrs
        )
        value = builder.repeat_interleave(
            value, repeats=num_repeats, dim=1, unit_attrs=unit_attrs
        )

    # Transpose key: [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
    key_transposed = builder.transpose(key, dim0=-2, dim1=-1)

    # Q @ K^T
    qk = builder.matmul(query, key_transposed, unit_attrs=unit_attrs)

    # Scale if provided
    if scale is not None:
        qk_shape = builder.get_shape(qk)
        scale_shape = [1] * len(qk_shape)
        scale_tensor = builder.full(
            scale_shape, torch.bfloat16, scale, unit_attrs=unit_attrs
        )
        qk = builder.multiply(qk, scale_tensor, unit_attrs=unit_attrs)

    # Add attention mask if provided
    if attention_mask is not None:
        qk = builder.add(qk, attention_mask, unit_attrs=unit_attrs)

    # Softmax on last dimension
    softmax_out = builder.softmax(qk, dimension=-1, unit_attrs=unit_attrs)

    # @ V
    output = builder.matmul(softmax_out, value, unit_attrs=unit_attrs)

    return output


@pytest.mark.parametrize(
    "shapes",
    [
        # LLaMA-7B style: 32 heads, 128 head_dim
        [
            (1, 32, 512, 128),
            (1, 32, 512, 128),
            (1, 32, 512, 128),
        ],
        # Mistral-7B GQA: 32 Q heads, 8 KV heads (4:1 ratio), 128 head_dim
        [
            (1, 32, 512, 128),
            (1, 8, 512, 128),
            (1, 8, 512, 128),
        ],
        # LLaMA-2-70B GQA: 64 Q heads, 8 KV heads (8:1 ratio), 128 head_dim
        [
            (1, 64, 256, 128),
            (1, 8, 256, 128),
            (1, 8, 256, 128),
        ],
        # BERT-base: 12 heads, 64 head_dim, seq=512
        [
            (1, 12, 512, 64),
            (1, 12, 512, 64),
            (1, 12, 512, 64),
        ],
        # ViT-B/16: 12 heads, 64 head_dim, seq=197 (196 patches + CLS token)
        [
            (1, 12, 197, 64),
            (1, 12, 197, 64),
            (1, 12, 197, 64),
        ],
        # GPT-2 small: 12 heads, 64 head_dim, seq=1024
        [
            (1, 12, 1024, 64),
            (1, 12, 1024, 64),
            (1, 12, 1024, 64),
        ],
        # Stable Diffusion cross-attention: different Q/K seq_lens (latent to text)
        [
            (1, 8, 4096, 64),
            (1, 8, 77, 64),
            (1, 8, 77, 64),
        ],
        # Phi-3 style GQA: 32 Q heads, 8 KV heads, 96 head_dim (not divisible by 32)
        [
            (1, 32, 256, 96),
            (1, 8, 256, 96),
            (1, 8, 256, 96),
        ],
    ],
)
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa(shapes: List[Shape], use_mask: bool, target: str, request, device):
    """
    Test Scaled Dot Product Attention pattern fusion (prefill) with optional mask.
    This test implements the SDPA operation as a sequence of TTIR ops:
    - Transpose key
    - Q @ K^T matmul
    - Scale by 1/sqrt(head_dim)
    - Add attention mask (optional, causal triu mask)
    - Softmax
    - @ V matmul

    Expected to fuse into ttnn.scaled_dot_product_attention
    """
    query_shape, key_shape, value_shape = shapes
    mask_shape = (query_shape[0], 1, query_shape[2], key_shape[2])
    input_shapes = shapes + [mask_shape] if use_mask else shapes
    dtypes = [torch.bfloat16] * len(input_shapes)

    def module_sdpa_no_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_no_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query_data = torch.randn(query_shape, dtype=torch.bfloat16)
            key_data = torch.randn(key_shape, dtype=torch.bfloat16)
            value_data = torch.randn(value_shape, dtype=torch.bfloat16)

            head_dim = query_shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

            golden_output = build_torch_golden(
                query_data, key_data, value_data, scale=scale
            )

            result = build_ttir(
                query, key, value, builder, scale=scale, unit_attrs=unit_attrs
            )

            builder.set_goldens(
                {query: query_data, key: key_data, value: value_data},
                {result: golden_output},
            )
            return result

    def module_sdpa_with_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_with_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query_data = torch.randn(query_shape, dtype=torch.bfloat16)
            key_data = torch.randn(key_shape, dtype=torch.bfloat16)
            value_data = torch.randn(value_shape, dtype=torch.bfloat16)
            mask_data = torch.triu(
                torch.full(mask_shape, float("-inf"), dtype=torch.bfloat16), diagonal=1
            )

            head_dim = query_shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

            golden_output = build_torch_golden(
                query_data, key_data, value_data, scale=scale, attention_mask=mask_data
            )

            result = build_ttir(
                query,
                key,
                value,
                builder,
                scale=scale,
                attention_mask=attention_mask,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens(
                {
                    query: query_data,
                    key: key_data,
                    value: value_data,
                    attention_mask: mask_data,
                },
                {result: golden_output},
            )
            return result

    output = compile_and_execute_ttir(
        module_sdpa_with_mask if use_mask else module_sdpa_no_mask,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "scaled_dot_product_attention")


@pytest.mark.parametrize(
    "shapes",
    [
        # LLaMA-7B decode: 32 heads, 128 head_dim, kv_seq=2048
        [
            (1, 32, 1, 128),
            (1, 32, 2048, 128),
            (1, 32, 2048, 128),
        ],
        # Mistral-7B GQA decode: 32 Q heads, 8 KV heads, 128 head_dim, kv_seq=2048
        [
            (1, 32, 1, 128),
            (1, 8, 2048, 128),
            (1, 8, 2048, 128),
        ],
        # LLaMA-2-70B GQA decode: 64 Q heads, 8 KV heads (8:1 ratio), 128 head_dim
        pytest.param(
            [
                (1, 64, 1, 128),
                (1, 8, 2048, 128),
                (1, 8, 2048, 128),
            ],
            marks=pytest.mark.xfail(
                reason="Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/33440"
            ),
        ),
        # GPT-2 decode: 12 heads, 64 head_dim, kv_seq=1024
        [
            (1, 12, 1, 64),
            (1, 12, 1024, 64),
            (1, 12, 1024, 64),
        ],
        # Batched decode for throughput: batch=32 (common for serving)
        [
            (32, 32, 1, 128),
            (32, 32, 512, 128),
            (32, 32, 512, 128),
        ],
        # Batched GQA decode: batch=8, Mistral-style
        [
            (8, 32, 1, 128),
            (8, 8, 1024, 128),
            (8, 8, 1024, 128),
        ],
        # Phi-3 style GQA decode: 32 Q heads, 8 KV heads, 96 head_dim (not divisible by 32)
        [
            (1, 32, 1, 96),
            (1, 8, 1024, 96),
            (1, 8, 1024, 96),
        ],
    ],
)
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decode(shapes: List[Shape], use_mask: bool, target: str, request, device):
    """
    Test Scaled Dot Product Attention pattern fusion for decode (query_seq_len=1).
    This test implements the SDPA operation as a sequence of TTIR ops:
    - Transpose key
    - Q @ K^T matmul
    - Scale by 1/sqrt(head_dim)
    - Add attention mask (optional, zeros mask for decode)
    - Softmax
    - @ V matmul

    Expected to fuse into ttnn.scaled_dot_product_attention
    """
    query_shape, key_shape, value_shape = shapes
    mask_shape = (query_shape[0], 1, query_shape[2], key_shape[2])
    input_shapes = shapes + [mask_shape] if use_mask else shapes
    dtypes = [torch.bfloat16] * len(input_shapes)

    def module_sdpa_no_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_no_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query_data = torch.randn(query_shape, dtype=torch.bfloat16)
            key_data = torch.randn(key_shape, dtype=torch.bfloat16)
            value_data = torch.randn(value_shape, dtype=torch.bfloat16)

            head_dim = query_shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

            golden_output = build_torch_golden(
                query_data, key_data, value_data, scale=scale
            )

            result = build_ttir(
                query, key, value, builder, scale=scale, unit_attrs=unit_attrs
            )

            builder.set_goldens(
                {query: query_data, key: key_data, value: value_data},
                {result: golden_output},
            )
            return result

    def module_sdpa_with_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_with_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query_data = torch.randn(query_shape, dtype=torch.bfloat16)
            key_data = torch.randn(key_shape, dtype=torch.bfloat16)
            value_data = torch.randn(value_shape, dtype=torch.bfloat16)
            # For decode, use zeros mask (no causal masking needed)
            mask_data = torch.zeros(mask_shape, dtype=torch.bfloat16)

            head_dim = query_shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

            golden_output = build_torch_golden(
                query_data, key_data, value_data, scale=scale, attention_mask=mask_data
            )

            result = build_ttir(
                query,
                key,
                value,
                builder,
                scale=scale,
                attention_mask=attention_mask,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens(
                {
                    query: query_data,
                    key: key_data,
                    value: value_data,
                    attention_mask: mask_data,
                },
                {result: golden_output},
            )
            return result

    output = compile_and_execute_ttir(
        module_sdpa_with_mask if use_mask else module_sdpa_no_mask,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "scaled_dot_product_attention_decode")


def build_ttir_llama32_1b_style(
    query: Operand,
    key: Operand,
    value: Operand,
    builder: TTIRBuilder,
    scale: Optional[float] = None,
    attention_mask: Optional[Operand] = None,
    unit_attrs: Optional[List[str]] = None,
):
    """
    Build TTIR representation of SDPA pattern matching LLaMA 3.2 1B style from
    llama_test_irs/llama_3_2_1b.mlir.

    Key characteristics:
    - Single scaling: scale applied to QK^T result (not split)
    - GQA broadcast pattern: reshape → broadcast → reshape (not repeat_interleave)
    - Manual softmax with f32 intermediate: typecast → max → subtract → exp → sum → div → typecast
    - No all-masked row handling (simpler softmax)
    """
    q_shape = builder.get_shape(query)
    k_shape = builder.get_shape(key)
    v_shape = builder.get_shape(value)

    batch = q_shape[0]
    q_heads = q_shape[1]
    q_seq = q_shape[2]
    head_dim = q_shape[3]
    kv_heads = k_shape[1]
    kv_seq = k_shape[2]

    # Handle GQA: broadcast K/V heads to match Q heads using reshape → broadcast → reshape
    if q_heads != kv_heads:
        assert (
            q_heads % kv_heads == 0
        ), f"Q heads ({q_heads}) must be divisible by KV heads ({kv_heads})"
        num_repeats = q_heads // kv_heads

        # For K: reshape to [batch, kv_heads, 1, kv_seq, head_dim]
        k_5d_shape = [batch, kv_heads, 1, kv_seq, head_dim]
        key_5d = builder.reshape(key, k_5d_shape, unit_attrs=unit_attrs)

        k_broadcast_factors = [1, 1, num_repeats, 1, 1]
        key_broadcast = builder.broadcast(
            key_5d, k_broadcast_factors, unit_attrs=unit_attrs
        )

        # Reshape back to [batch, q_heads, kv_seq, head_dim]
        k_final_shape = [batch, q_heads, kv_seq, head_dim]
        key = builder.reshape(key_broadcast, k_final_shape, unit_attrs=unit_attrs)

        # Same for V
        v_5d_shape = [batch, kv_heads, 1, kv_seq, head_dim]
        value_5d = builder.reshape(value, v_5d_shape, unit_attrs=unit_attrs)

        v_broadcast_factors = [1, 1, num_repeats, 1, 1]
        value_broadcast = builder.broadcast(
            value_5d, v_broadcast_factors, unit_attrs=unit_attrs
        )

        v_final_shape = [batch, q_heads, kv_seq, head_dim]
        value = builder.reshape(value_broadcast, v_final_shape, unit_attrs=unit_attrs)

    # Transpose key: [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
    key_transposed = builder.transpose(key, dim0=-2, dim1=-1)

    # Q @ K^T
    qk = builder.matmul(query, key_transposed, unit_attrs=unit_attrs)

    # Scale if provided (single scaling on QK^T result)
    if scale is not None:
        qk_shape = builder.get_shape(qk)
        scale_shape = [1] * len(qk_shape)
        scale_tensor = builder.full(
            scale_shape, torch.bfloat16, scale, unit_attrs=unit_attrs
        )
        qk = builder.multiply(qk, scale_tensor, unit_attrs=unit_attrs)

    # Add attention mask if provided
    if attention_mask is not None:
        qk = builder.add(qk, attention_mask, unit_attrs=unit_attrs)

    # Manual softmax with f32 intermediate (matching llama_3_2_1b.mlir pattern)
    qk_shape = builder.get_shape(qk)

    # Typecast to f32
    qk_f32 = builder.typecast(qk, torch.float32, unit_attrs=unit_attrs)

    # Max along last dimension
    qk_max = builder.max(qk_f32, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs)
    qk_max_shape = builder.get_shape(qk_max)
    qk_max_4d = builder.reshape(qk_max, list(qk_max_shape) + [1], unit_attrs=unit_attrs)
    qk_max_broadcast_factors = [1, 1, 1, qk_shape[-1]]
    qk_max_broadcast = builder.broadcast(
        qk_max_4d, qk_max_broadcast_factors, unit_attrs=unit_attrs
    )

    # Subtract max for numerical stability
    qk_shifted = builder.subtract(qk_f32, qk_max_broadcast, unit_attrs=unit_attrs)

    # Exp
    qk_exp = builder.exp(qk_shifted, unit_attrs=unit_attrs)

    # Sum along last dimension
    qk_sum = builder.sum(qk_exp, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs)
    qk_sum_shape = builder.get_shape(qk_sum)
    qk_sum_4d = builder.reshape(qk_sum, list(qk_sum_shape) + [1], unit_attrs=unit_attrs)
    qk_sum_broadcast_factors = [1, 1, 1, qk_shape[-1]]
    qk_sum_broadcast = builder.broadcast(
        qk_sum_4d, qk_sum_broadcast_factors, unit_attrs=unit_attrs
    )

    # Divide
    softmax_f32 = builder.div(qk_exp, qk_sum_broadcast, unit_attrs=unit_attrs)

    # Typecast back to bf16
    softmax_out = builder.typecast(softmax_f32, torch.bfloat16, unit_attrs=unit_attrs)

    # @ V
    output = builder.matmul(softmax_out, value, unit_attrs=unit_attrs)

    return output


def build_ttir_llama32_style(
    query: Operand,
    key: Operand,
    value: Operand,
    builder: TTIRBuilder,
    scale: Optional[float] = None,
    attention_mask: Optional[Operand] = None,
    unit_attrs: Optional[List[str]] = None,
):
    """
    Build TTIR representation of SDPA pattern matching LLaMA 3.2 style from
    llama_test_irs/llama_3_2_1b_1_layer_new_version.mlir.

    Key differences from standard SDPA:
    - Split scaling: scale applied to both Q and K^T separately (not just QK^T)
    - GQA broadcast pattern: reshape → broadcast → reshape (not repeat_interleave)
    - Complex softmax with all-masked row handling
    - F32 intermediate computation
    """
    q_shape = builder.get_shape(query)
    k_shape = builder.get_shape(key)
    v_shape = builder.get_shape(value)

    batch = q_shape[0]
    q_heads = q_shape[1]
    q_seq = q_shape[2]
    head_dim = q_shape[3]
    kv_heads = k_shape[1]
    kv_seq = k_shape[2]

    # Cast Q to f32 and apply scale
    query_f32 = builder.typecast(query, torch.float32, unit_attrs=unit_attrs)
    if scale is not None:
        scale_q_shape = [1] * len(q_shape)
        scale_q_data = torch.full(scale_q_shape, scale, dtype=torch.float32)
        scale_q_tensor = builder.constant(scale_q_data, unit_attrs=unit_attrs)
        query_f32 = builder.multiply(query_f32, scale_q_tensor, unit_attrs=unit_attrs)

    # Handle GQA: broadcast K/V heads to match Q heads using reshape → broadcast → reshape
    if q_heads != kv_heads:
        assert (
            q_heads % kv_heads == 0
        ), f"Q heads ({q_heads}) must be divisible by KV heads ({kv_heads})"
        num_repeats = q_heads // kv_heads

        # For K: reshape to [batch, kv_heads, 1, kv_seq, head_dim]
        k_5d_shape = [batch, kv_heads, 1, kv_seq, head_dim]
        key_5d = builder.reshape(key, k_5d_shape, unit_attrs=unit_attrs)

        k_broadcast_factors = [1, 1, num_repeats, 1, 1]
        key_broadcast = builder.broadcast(
            key_5d, k_broadcast_factors, unit_attrs=unit_attrs
        )

        # Reshape back to [batch, q_heads, kv_seq, head_dim]
        k_final_shape = [batch, q_heads, kv_seq, head_dim]
        key = builder.reshape(key_broadcast, k_final_shape, unit_attrs=unit_attrs)

        # Same for V
        v_5d_shape = [batch, kv_heads, 1, kv_seq, head_dim]
        value_5d = builder.reshape(value, v_5d_shape, unit_attrs=unit_attrs)

        v_broadcast_factors = [1, 1, num_repeats, 1, 1]
        value_broadcast = builder.broadcast(
            value_5d, v_broadcast_factors, unit_attrs=unit_attrs
        )

        v_final_shape = [batch, q_heads, kv_seq, head_dim]
        value = builder.reshape(value_broadcast, v_final_shape, unit_attrs=unit_attrs)

    # Cast K to f32, transpose, and apply scale
    key_f32 = builder.typecast(key, torch.float32, unit_attrs=unit_attrs)
    key_transposed = builder.transpose(key_f32, dim0=-2, dim1=-1)

    if scale is not None:
        kt_shape = builder.get_shape(key_transposed)
        scale_k_shape = [1] * len(kt_shape)
        scale_k_data = torch.full(scale_k_shape, scale, dtype=torch.float32)
        scale_k_tensor = builder.constant(scale_k_data, unit_attrs=unit_attrs)
        key_transposed = builder.multiply(
            key_transposed, scale_k_tensor, unit_attrs=unit_attrs
        )

    # Q @ K^T in f32
    qk = builder.matmul(query_f32, key_transposed, unit_attrs=unit_attrs)

    # Add attention mask if provided
    if attention_mask is not None:
        mask_f32 = builder.typecast(
            attention_mask, torch.float32, unit_attrs=unit_attrs
        )
        qk = builder.add(qk, mask_f32, unit_attrs=unit_attrs)

    # Softmax with all-masked row handling
    # Check for all -inf rows (all-masked)
    qk_shape = builder.get_shape(qk)
    neg_inf_data = torch.full(qk_shape, float("-inf"), dtype=torch.float32)
    neg_inf_tensor = builder.constant(neg_inf_data, unit_attrs=unit_attrs)
    is_neg_inf = builder.eq(qk, neg_inf_tensor, unit_attrs=unit_attrs)
    is_not_neg_inf = builder.logical_not(is_neg_inf, unit_attrs=unit_attrs)

    # Check if any element in each row is not -inf
    any_valid = builder.reduce_or(
        is_not_neg_inf, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs
    )
    any_valid_shape = builder.get_shape(any_valid)
    any_valid_4d = builder.reshape(
        any_valid, any_valid_shape + [1], unit_attrs=unit_attrs
    )

    all_masked = builder.logical_not(any_valid_4d, unit_attrs=unit_attrs)
    all_masked_broadcast_factors = [1, 1, 1, qk_shape[-1]]
    all_masked_broadcast = builder.broadcast(
        all_masked, all_masked_broadcast_factors, unit_attrs=unit_attrs
    )

    # Standard softmax: max → subtract → exp → sum → div
    qk_max = builder.max(qk, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs)
    qk_max_shape = builder.get_shape(qk_max)
    qk_max_4d = builder.reshape(qk_max, list(qk_max_shape) + [1], unit_attrs=unit_attrs)
    qk_max_broadcast_factors = [1, 1, 1, qk_shape[-1]]
    qk_max_broadcast = builder.broadcast(
        qk_max_4d, qk_max_broadcast_factors, unit_attrs=unit_attrs
    )

    qk_shifted = builder.subtract(qk, qk_max_broadcast, unit_attrs=unit_attrs)
    qk_exp = builder.exp(qk_shifted, unit_attrs=unit_attrs)

    qk_sum = builder.sum(qk_exp, dim_arg=[-1], keep_dim=False, unit_attrs=unit_attrs)
    qk_sum_shape = builder.get_shape(qk_sum)
    qk_sum_4d = builder.reshape(qk_sum, list(qk_sum_shape) + [1], unit_attrs=unit_attrs)
    qk_sum_broadcast_factors = [1, 1, 1, qk_shape[-1]]
    qk_sum_broadcast = builder.broadcast(
        qk_sum_4d, qk_sum_broadcast_factors, unit_attrs=unit_attrs
    )

    softmax_out = builder.div(qk_exp, qk_sum_broadcast, unit_attrs=unit_attrs)

    # Replace all-masked rows with zeros
    zeros_data = torch.zeros(qk_shape, dtype=torch.float32)
    zeros = builder.constant(zeros_data, unit_attrs=unit_attrs)
    softmax_out = builder.where(
        all_masked_broadcast, zeros, softmax_out, unit_attrs=unit_attrs
    )

    # Cast V to f32 and compute @ V
    value_f32 = builder.typecast(value, torch.float32, unit_attrs=unit_attrs)
    output_f32 = builder.matmul(softmax_out, value_f32, unit_attrs=unit_attrs)

    # Cast back to bf16
    output = builder.typecast(output_f32, torch.bfloat16, unit_attrs=unit_attrs)

    return output


@pytest.mark.parametrize(
    "shapes",
    [
        # LLaMA 3.2 1B decode style: batch=32, q_heads=32, kv_heads=8, q_seq=1, kv_seq=128, head_dim=64
        [
            (32, 32, 1, 64),  # Q: [batch, q_heads, q_seq, head_dim]
            (32, 8, 128, 64),  # K: [batch, kv_heads, kv_seq, head_dim]
            (32, 8, 128, 64),  # V: [batch, kv_heads, kv_seq, head_dim]
        ],
        # LLaMA 3.2 1B with smaller batch
        [
            (1, 32, 1, 64),
            (1, 8, 128, 64),
            (1, 8, 128, 64),
        ],
        # LLaMA 3.2 with longer kv_seq
        [
            (8, 32, 1, 64),
            (8, 8, 512, 64),
            (8, 8, 512, 64),
        ],
    ],
)
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_llama32_style(
    shapes: List[Shape], use_mask: bool, target: str, request, device
):
    """
    Test SDPA pattern fusion matching LLaMA 3.2 1B style from
    llama_test_irs/llama_3_2_1b_1_layer_new_version.mlir.

    Key characteristics:
    - Split scaling: scale applied to both Q and K^T (1/sqrt(head_dim) each)
    - GQA using reshape → broadcast → reshape pattern
    - F32 intermediate computations
    - Complex softmax with all-masked row handling

    Expected to fuse into ttnn.scaled_dot_product_attention_decode
    """
    query_shape, key_shape, value_shape = shapes
    mask_shape = (query_shape[0], 1, query_shape[2], key_shape[2])
    input_shapes = shapes + [mask_shape] if use_mask else shapes
    dtypes = [torch.bfloat16] * len(input_shapes)

    def module_sdpa_no_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_llama32_no_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query_data = torch.randn(query_shape, dtype=torch.bfloat16)
            key_data = torch.randn(key_shape, dtype=torch.bfloat16)
            value_data = torch.randn(value_shape, dtype=torch.bfloat16)

            head_dim = query_shape[-1]
            # Split scale: sqrt(1/head_dim) applied to both Q and K
            scale = 1.0 / math.sqrt(math.sqrt(head_dim))

            # For golden, use the combined scale (1/sqrt(head_dim))
            combined_scale = 1.0 / math.sqrt(head_dim)
            golden_output = build_torch_golden(
                query_data, key_data, value_data, scale=combined_scale
            )

            result = build_ttir_llama32_style(
                query, key, value, builder, scale=scale, unit_attrs=unit_attrs
            )

            builder.set_goldens(
                {query: query_data, key: key_data, value: value_data},
                {result: golden_output},
            )
            return result

    def module_sdpa_with_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_llama32_with_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query_data = torch.randn(query_shape, dtype=torch.bfloat16)
            key_data = torch.randn(key_shape, dtype=torch.bfloat16)
            value_data = torch.randn(value_shape, dtype=torch.bfloat16)
            # Causal mask for decode: positions after current are masked
            mask_data = torch.zeros(mask_shape, dtype=torch.bfloat16)

            head_dim = query_shape[-1]
            scale = 1.0 / math.sqrt(math.sqrt(head_dim))

            combined_scale = 1.0 / math.sqrt(head_dim)
            golden_output = build_torch_golden(
                query_data,
                key_data,
                value_data,
                scale=combined_scale,
                attention_mask=mask_data,
            )

            result = build_ttir_llama32_style(
                query,
                key,
                value,
                builder,
                scale=scale,
                attention_mask=attention_mask,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens(
                {
                    query: query_data,
                    key: key_data,
                    value: value_data,
                    attention_mask: mask_data,
                },
                {result: golden_output},
            )
            return result

    output = compile_and_execute_ttir(
        module_sdpa_with_mask if use_mask else module_sdpa_no_mask,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "scaled_dot_product_attention_decode")


@pytest.mark.parametrize(
    "shapes",
    [
        # LLaMA 3.2 1B decode style: batch=32, q_heads=32, kv_heads=8, q_seq=1, kv_seq=128, head_dim=64
        [
            (32, 32, 1, 64),  # Q: [batch, q_heads, q_seq, head_dim]
            (32, 8, 128, 64),  # K: [batch, kv_heads, kv_seq, head_dim]
            (32, 8, 128, 64),  # V: [batch, kv_heads, kv_seq, head_dim]
        ],
        # LLaMA 3.2 1B with smaller batch
        [
            (1, 32, 1, 64),
            (1, 8, 128, 64),
            (1, 8, 128, 64),
        ],
        # LLaMA 3.2 with longer kv_seq
        [
            (8, 32, 1, 64),
            (8, 8, 512, 64),
            (8, 8, 512, 64),
        ],
        # Non-GQA variant (same heads for Q/K/V)
        [
            (1, 32, 1, 64),
            (1, 32, 128, 64),
            (1, 32, 128, 64),
        ],
    ],
)
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_llama32_1b_style(
    shapes: List[Shape], use_mask: bool, target: str, request, device
):
    """
    Test SDPA pattern fusion matching LLaMA 3.2 1B style from
    llama_test_irs/llama_3_2_1b.mlir.

    Key characteristics:
    - Single scaling: scale applied to QK^T result (1/sqrt(head_dim))
    - GQA using reshape → broadcast → reshape pattern
    - Manual softmax with f32 intermediate: typecast → max → subtract → exp → sum → div → typecast
    - No all-masked row handling (simpler softmax than llama32_style)

    Expected to fuse into ttnn.scaled_dot_product_attention_decode
    """
    query_shape, key_shape, value_shape = shapes
    mask_shape = (query_shape[0], 1, query_shape[2], key_shape[2])
    input_shapes = shapes + [mask_shape] if use_mask else shapes
    dtypes = [torch.bfloat16] * len(input_shapes)

    def module_sdpa_no_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_llama32_1b_no_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query_data = torch.randn(query_shape, dtype=torch.bfloat16)
            key_data = torch.randn(key_shape, dtype=torch.bfloat16)
            value_data = torch.randn(value_shape, dtype=torch.bfloat16)

            head_dim = query_shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

            golden_output = build_torch_golden(
                query_data, key_data, value_data, scale=scale
            )

            result = build_ttir_llama32_1b_style(
                query, key, value, builder, scale=scale, unit_attrs=unit_attrs
            )

            builder.set_goldens(
                {query: query_data, key: key_data, value: value_data},
                {result: golden_output},
            )
            return result

    def module_sdpa_with_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_llama32_1b_with_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query_data = torch.randn(query_shape, dtype=torch.bfloat16)
            key_data = torch.randn(key_shape, dtype=torch.bfloat16)
            value_data = torch.randn(value_shape, dtype=torch.bfloat16)
            # Causal mask for decode: positions after current are masked
            mask_data = torch.zeros(mask_shape, dtype=torch.bfloat16)

            head_dim = query_shape[-1]
            scale = 1.0 / math.sqrt(head_dim)

            golden_output = build_torch_golden(
                query_data, key_data, value_data, scale=scale, attention_mask=mask_data
            )

            result = build_ttir_llama32_1b_style(
                query,
                key,
                value,
                builder,
                scale=scale,
                attention_mask=attention_mask,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens(
                {
                    query: query_data,
                    key: key_data,
                    value: value_data,
                    attention_mask: mask_data,
                },
                {result: golden_output},
            )
            return result

    output = compile_and_execute_ttir(
        module_sdpa_with_mask if use_mask else module_sdpa_no_mask,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "scaled_dot_product_attention_decode")
