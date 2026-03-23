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
from conftest import get_request_kwargs

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


def build_ttir_single_scale_simple_softmax(
    query: Operand,
    key: Operand,
    value: Operand,
    builder: TTIRBuilder,
    scale: Optional[float] = None,
    attention_mask: Optional[Operand] = None,
    unit_attrs: Optional[List[str]] = None,
):
    """
    Build TTIR representation of SDPA with single scaling and simple softmax.

    Pattern: Q @ K^T → scale → [mask] → softmax → @ V

    Key characteristics:
    - Single scaling: scale applied to QK^T result (not split across Q and K)
    - GQA broadcast pattern: reshape → broadcast → reshape
    - Simple softmax: typecast → max → subtract → exp → sum → div → typecast
    - No all-masked row handling
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


def build_ttir_split_scale_robust_softmax(
    query: Operand,
    key: Operand,
    value: Operand,
    builder: TTIRBuilder,
    attention_mask: Operand,
    scale: Optional[float] = None,
    unit_attrs: Optional[List[str]] = None,
):
    """
    Build TTIR representation of SDPA with split scaling and robust softmax.

    Pattern: (Q * scale) @ (K^T * scale) → mask → robust_softmax → @ V

    Key characteristics:
    - Split scaling: scale applied to both Q and K^T separately
    - GQA broadcast pattern: reshape → broadcast → reshape
    - Robust softmax with all-masked row handling (eq, logical_not, reduce_or, where)
    - Full f32 intermediate computation throughout
    - Requires attention mask (robust softmax is designed for masked scenarios)
    """
    assert attention_mask is not None, "Robust softmax pattern requires attention mask"
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

    # Add attention mask
    mask_f32 = builder.typecast(attention_mask, torch.float32, unit_attrs=unit_attrs)
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
        # GQA decode: batch=32, q_heads=32, kv_heads=8, q_seq=1, kv_seq=128, head_dim=64
        [
            (32, 32, 1, 64),  # Q: [batch, q_heads, q_seq, head_dim]
            (32, 8, 128, 64),  # K: [batch, kv_heads, kv_seq, head_dim]
            (32, 8, 128, 64),  # V: [batch, kv_heads, kv_seq, head_dim]
        ],
        # GQA decode with smaller batch
        [
            (1, 32, 1, 64),
            (1, 8, 128, 64),
            (1, 8, 128, 64),
        ],
        # GQA decode with longer kv_seq
        [
            (8, 32, 1, 64),
            (8, 8, 512, 64),
            (8, 8, 512, 64),
        ],
        # Non-GQA / MHA variant (same heads for Q/K/V)
        [
            (1, 32, 1, 64),
            (1, 32, 128, 64),
            (1, 32, 128, 64),
        ],
        # MQA decode (kv_heads=1, used in Falcon-style models)
        [
            (8, 32, 1, 64),
            (8, 1, 128, 64),
            (8, 1, 128, 64),
        ],
        # GQA with larger head_dim=128 (common in larger models)
        [
            (4, 64, 1, 128),
            (4, 8, 256, 128),
            (4, 8, 256, 128),
        ],
        # Prefill mode: q_seq > 1 (initial prompt processing)
        [
            (1, 32, 128, 64),  # Q: [batch, q_heads, q_seq=128, head_dim]
            (1, 8, 128, 64),  # K: [batch, kv_heads, kv_seq=128, head_dim]
            (1, 8, 128, 64),  # V: [batch, kv_heads, kv_seq=128, head_dim]
        ],
        # Prefill mode with longer sequence
        [
            (1, 32, 512, 64),
            (1, 8, 512, 64),
            (1, 8, 512, 64),
        ],
        # MHA prefill (non-GQA)
        [
            (1, 32, 128, 64),
            (1, 32, 128, 64),
            (1, 32, 128, 64),
        ],
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="The SDPA fusing pattern is only executed when the optimizer is enabled, but then the golden test fails: https://github.com/tenstorrent/tt-mlir/issues/5283"
)
def test_sdpa_split_scale_robust_softmax(
    shapes: List[Shape], target: str, request, device
):
    """
    Test SDPA pattern fusion with split scaling and robust softmax.

    Pattern: (Q * scale) @ (K^T * scale) → mask → robust_softmax → @ V

    Key characteristics:
    - Split scaling: scale applied to both Q and K^T (sqrt(1/head_dim) each)
    - GQA using reshape → broadcast → reshape pattern
    - Robust softmax with all-masked row handling
    - Always uses attention mask (robust softmax is designed for masked scenarios)

    Expected to fuse into ttnn.scaled_dot_product_attention_decode
    """
    query_shape, key_shape, value_shape = shapes
    mask_shape = (query_shape[0], 1, query_shape[2], key_shape[2])
    input_shapes = shapes + [mask_shape]
    dtypes = [torch.bfloat16] * len(input_shapes)

    def module_sdpa(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_split_scale_robust(
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
            mask_data = torch.zeros(mask_shape, dtype=torch.bfloat16)

            head_dim = query_shape[-1]
            # Split scale: sqrt(1/head_dim) applied to both Q and K
            scale = 1.0 / math.sqrt(math.sqrt(head_dim))

            # For golden, use the combined scale (1/sqrt(head_dim))
            combined_scale = 1.0 / math.sqrt(head_dim)
            golden_output = build_torch_golden(
                query_data,
                key_data,
                value_data,
                scale=combined_scale,
                attention_mask=mask_data,
            )

            result = build_ttir_split_scale_robust_softmax(
                query,
                key,
                value,
                builder,
                attention_mask,
                scale=scale,
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
        module_sdpa,
        target=target,
        **get_request_kwargs(request),
        device=device,
        compile_options=["enable-optimizer=true"],
    )

    # Check for appropriate fused op based on q_seq
    q_seq = query_shape[2]
    if q_seq == 1:
        assert check_op(output, "scaled_dot_product_attention_decode")
    else:
        assert check_op(output, "scaled_dot_product_attention")


@pytest.mark.parametrize(
    "shapes",
    [
        # GQA decode: batch=32, q_heads=32, kv_heads=8, q_seq=1, kv_seq=128, head_dim=64
        [
            (32, 32, 1, 64),  # Q: [batch, q_heads, q_seq, head_dim]
            (32, 8, 128, 64),  # K: [batch, kv_heads, kv_seq, head_dim]
            (32, 8, 128, 64),  # V: [batch, kv_heads, kv_seq, head_dim]
        ],
        # GQA decode with smaller batch
        [
            (1, 32, 1, 64),
            (1, 8, 128, 64),
            (1, 8, 128, 64),
        ],
        # GQA decode with longer kv_seq
        [
            (8, 32, 1, 64),
            (8, 8, 512, 64),
            (8, 8, 512, 64),
        ],
        # Non-GQA / MHA variant (same heads for Q/K/V)
        [
            (1, 32, 1, 64),
            (1, 32, 128, 64),
            (1, 32, 128, 64),
        ],
        # MQA decode (kv_heads=1, used in Falcon-style models)
        [
            (8, 32, 1, 64),
            (8, 1, 128, 64),
            (8, 1, 128, 64),
        ],
        # GQA with larger head_dim=128 (common in larger models)
        [
            (4, 64, 1, 128),
            (4, 8, 256, 128),
            (4, 8, 256, 128),
        ],
        # Prefill mode: q_seq > 1 (initial prompt processing)
        [
            (1, 32, 128, 64),  # Q: [batch, q_heads, q_seq=128, head_dim]
            (1, 8, 128, 64),  # K: [batch, kv_heads, kv_seq=128, head_dim]
            (1, 8, 128, 64),  # V: [batch, kv_heads, kv_seq=128, head_dim]
        ],
        # Prefill mode with longer sequence
        [
            (1, 32, 512, 64),
            (1, 8, 512, 64),
            (1, 8, 512, 64),
        ],
        # MHA prefill (non-GQA)
        [
            (1, 32, 128, 64),
            (1, 32, 128, 64),
            (1, 32, 128, 64),
        ],
    ],
)
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="The SDPA fusing pattern is only executed when the optimizer is enabled, but then the golden test fails: https://github.com/tenstorrent/tt-mlir/issues/5283"
)
def test_sdpa_single_scale_simple_softmax(
    shapes: List[Shape], use_mask: bool, target: str, request, device
):
    """
    Test SDPA pattern fusion with single scaling and simple softmax.

    Pattern: Q @ K^T → scale → [mask] → softmax → @ V

    Key characteristics:
    - Single scaling: scale applied to QK^T result (1/sqrt(head_dim))
    - GQA using reshape → broadcast → reshape pattern
    - Simple softmax: typecast → max → subtract → exp → sum → div → typecast
    - No all-masked row handling
    - Supports both decode (q_seq=1) and prefill (q_seq>1) modes

    Expected to fuse into:
    - ttnn.scaled_dot_product_attention_decode (when q_seq=1)
    - ttnn.scaled_dot_product_attention (when q_seq>1)
    """
    query_shape, key_shape, value_shape = shapes
    mask_shape = (query_shape[0], 1, query_shape[2], key_shape[2])
    input_shapes = shapes + [mask_shape] if use_mask else shapes
    dtypes = [torch.bfloat16] * len(input_shapes)

    def module_sdpa_no_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_single_scale_simple_no_mask(
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

            result = build_ttir_single_scale_simple_softmax(
                query, key, value, builder, scale=scale, unit_attrs=unit_attrs
            )

            builder.set_goldens(
                {query: query_data, key: key_data, value: value_data},
                {result: golden_output},
            )
            return result

    def module_sdpa_with_mask(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_single_scale_simple_with_mask(
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

            result = build_ttir_single_scale_simple_softmax(
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
        **get_request_kwargs(request),
        device=device,
        compile_options=["enable-optimizer=true"],
    )

    # Check for appropriate fused op based on q_seq
    q_seq = query_shape[2]
    if q_seq == 1:
        assert check_op(output, "scaled_dot_product_attention_decode")
    else:
        assert check_op(output, "scaled_dot_product_attention")
