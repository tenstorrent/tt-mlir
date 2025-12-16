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

    assert check_op(output, "scaled_dot_product_attention")
