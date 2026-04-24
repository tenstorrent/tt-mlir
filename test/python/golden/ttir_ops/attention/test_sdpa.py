# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "shapes,scale",
    [
        # --- No GQA (Q heads == KV heads) ---
        # BERT: batch=8, heads=12, seq=384, head_dim=64
        (
            [(8, 12, 384, 64), (8, 12, 384, 64), (8, 12, 384, 64)],
            0.124999993,
        ),
        # ViT: batch=8, heads=12, seq=197, head_dim=64
        (
            [(8, 12, 197, 64), (8, 12, 197, 64), (8, 12, 197, 64)],
            0.124999993,
        ),
        # Phi-1/1.5: batch=32, heads=32, q_seq=17, kv_seq=128, head_dim=64
        (
            [(32, 32, 17, 64), (32, 32, 128, 64), (32, 32, 128, 64)],
            0.124999993,
        ),
        # Phi-2: batch=32, heads=32, q_seq=17, kv_seq=128, head_dim=80 (not 32-aligned)
        (
            [(32, 32, 17, 80), (32, 32, 128, 80), (32, 32, 128, 80)],
            0.111803405,
        ),
        # --- GQA (Q heads != KV heads) ---
        # Llama 3.2 1B: heads=32/8 (4:1 GQA), q_seq=18, kv_seq=128, head_dim=64
        (
            [(32, 32, 18, 64), (32, 8, 128, 64), (32, 8, 128, 64)],
            0.124999993,
        ),
        # Llama 3.1 8B: heads=32/8 (4:1 GQA), q_seq=18, kv_seq=128, head_dim=128
        (
            [(32, 32, 18, 128), (32, 8, 128, 128), (32, 8, 128, 128)],
            0.0883883387,
        ),
        # Llama 3.2 3B: heads=24/8 (3:1 GQA), q_seq=18, kv_seq=128, head_dim=128
        (
            [(32, 24, 18, 128), (32, 8, 128, 128), (32, 8, 128, 128)],
            0.0883883387,
        ),
        # Qwen 2.5 0.5B: heads=14/2 (7:1 GQA), q_seq=17, kv_seq=128, head_dim=64
        (
            [(32, 14, 17, 64), (32, 2, 128, 64), (32, 2, 128, 64)],
            0.124999993,
        ),
        # Qwen 2.5 1.5B: heads=12/2 (6:1 GQA), q_seq=17, kv_seq=128, head_dim=128
        (
            [(32, 12, 17, 128), (32, 2, 128, 128), (32, 2, 128, 128)],
            0.0883883387,
        ),
        # Qwen 2.5 3B: heads=16/2 (8:1 GQA), q_seq=17, kv_seq=128, head_dim=128
        (
            [(32, 16, 17, 128), (32, 2, 128, 128), (32, 2, 128, 128)],
            0.0883883387,
        ),
        # Qwen 2.5 7B: heads=28/4 (7:1 GQA), q_seq=17, kv_seq=128, head_dim=128
        (
            [(32, 28, 17, 128), (32, 4, 128, 128), (32, 4, 128, 128)],
            0.0883883387,
        ),
        # Qwen 3 0.6B/1.7B: heads=16/8 (2:1 GQA), q_seq=17, kv_seq=128, head_dim=128
        (
            [(32, 16, 17, 128), (32, 8, 128, 128), (32, 8, 128, 128)],
            0.0883883387,
        ),
        # Qwen 3 4B/8B: heads=32/8 (4:1 GQA), q_seq=17, kv_seq=128, head_dim=128
        (
            [(32, 32, 17, 128), (32, 8, 128, 128), (32, 8, 128, 128)],
            0.0883883387,
        ),
        # Qwen 3 Embedding 4B: heads=32/8 (4:1 GQA), q_seq=18, kv_seq=18, head_dim=128
        (
            [(32, 32, 18, 128), (32, 8, 18, 128), (32, 8, 18, 128)],
            0.0883883387,
        ),
        # Falcon 3.1B: heads=8/4 (2:1 GQA), q_seq=17, kv_seq=128, head_dim=256
        (
            [(32, 8, 17, 256), (32, 4, 128, 256), (32, 4, 128, 256)],
            6.250000,
        ),
        # Falcon 3.3B: heads=12/4 (3:1 GQA), q_seq=17, kv_seq=128, head_dim=256
        (
            [(32, 12, 17, 256), (32, 4, 128, 256), (32, 4, 128, 256)],
            6.250000,
        ),
        # Gemma 1.1 2B: heads=8/1 (8:1 GQA), q_seq=17, kv_seq=128, head_dim=256
        (
            [(32, 8, 17, 256), (32, 1, 128, 256), (32, 1, 128, 256)],
            6.250000,
        ),
        # Gemma 2 2B: heads=8/4 (2:1 GQA), q_seq=17, kv_seq=128, head_dim=256
        (
            [(32, 8, 17, 256), (32, 4, 128, 256), (32, 4, 128, 256)],
            6.250000,
        ),
        # --- Large sequence length disparity ---
        # SegFormer (block): heads=1, q_seq=256, kv_seq=4, head_dim=32
        (
            [(1, 1, 256, 32), (1, 1, 4, 32), (1, 1, 4, 32)],
            0.176795587,
        ),
        # SegFormer (layer): heads=1, q_seq=16384, kv_seq=256, head_dim=32
        (
            [(1, 1, 16384, 32), (1, 1, 256, 32), (1, 1, 256, 32)],
            0.176795587,
        ),
    ],
    ids=[
        "bert",
        "vit",
        "phi_1",
        "phi_2",
        "llama_3_2_1b",
        "llama_3_1_8b",
        "llama_3_2_3b",
        "qwen_2_5_0_5b",
        "qwen_2_5_1_5b",
        "qwen_2_5_3b",
        "qwen_2_5_7b",
        "qwen_3_0_6b",
        "qwen_3_4b",
        "qwen_3_embed_4b",
        "falcon_3_1b",
        "falcon_3_3b",
        "gemma_1_1_2b",
        "gemma_2_2b",
        "segformer_block",
        "segformer_layer",
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_in_models(
    shapes: List[Shape],
    scale: float,
    target: str,
    request,
    device,
):
    """
    Test Scaled Dot Product Attention with real model shapes extracted from
    single_blocks_and_layers model tests after TTNNFusing.
    """
    dtypes = [torch.bfloat16] * len(shapes)

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def sdpa(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes,mask_shape,scale",
    [
        # mask[0]=1 (broadcast batch), mask[1]=1 (broadcast heads)
        (
            [(1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)],
            (1, 1, 128, 128),
            0.124999993,
        ),
        # mask[0]=batch, mask[1]=num_heads (no broadcast)
        (
            [(1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)],
            (1, 8, 128, 128),
            0.124999993,
        ),
        # mask[0]=1 (broadcast batch), mask[1]=num_heads
        (
            [(2, 8, 128, 64), (2, 8, 128, 64), (2, 8, 128, 64)],
            (1, 8, 128, 128),
            0.124999993,
        ),
        # GQA: mask[1]=num_query_heads (not kv_heads)
        (
            [(1, 32, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)],
            (1, 32, 128, 128),
            0.124999993,
        ),
    ],
    ids=[
        "mask_broadcast_batch_and_heads",
        "mask_full_heads",
        "mask_broadcast_batch_full_heads",
        "mask_gqa_full_query_heads",
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_mask_broadcast(
    shapes: List[Shape],
    mask_shape: Shape,
    scale: float,
    target: str,
    request,
    device,
):
    """
    Test Scaled Dot Product Attention with different mask broadcast shapes.

    tt-metal supports mask shapes where batch and heads dimensions can be
    either 1 (broadcast) or match the query dimensions.
    """
    all_shapes = shapes + [mask_shape]
    dtypes = [torch.bfloat16] * len(all_shapes)

    def module(builder: TTIRBuilder):
        @builder.func(all_shapes, dtypes)
        def sdpa_mask_broadcast(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes,mask_shape,scale",
    [
        # Decode mask: [batch, 1, num_heads, kv_seq] - standard
        pytest.param(
            [(1, 32, 32, 64), (32, 32, 128, 64), (32, 32, 128, 64), (32,)],
            (32, 1, 32, 128),
            0.124999993,
            marks=pytest.mark.xfail(
                reason="PCC regression, tracked in "
                "https://github.com/tenstorrent/tt-mlir/issues/8112"
            ),
        ),
        # Decode mask with GQA: mask num_heads matches query num_heads
        pytest.param(
            [(1, 32, 32, 64), (32, 8, 128, 64), (32, 8, 128, 64), (32,)],
            (32, 1, 32, 128),
            0.124999993,
            marks=pytest.mark.xfail(
                reason="PCC regression, tracked in "
                "https://github.com/tenstorrent/tt-mlir/issues/8112"
            ),
        ),
        # Decode mask with batch broadcast: mask[0]=1
        (
            [(1, 32, 32, 64), (32, 32, 128, 64), (32, 32, 128, 64), (32,)],
            (1, 1, 32, 128),
            0.0883883387,
        ),
    ],
    ids=[
        "decode_mask_standard",
        "decode_mask_gqa",
        "decode_mask_broadcast_batch",
    ],
)
@pytest.mark.skip_exec(
    ("p150",),
    reason="SDPA decode kernel exceeds max runtime args on p150 (344 > 341). ",
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decode_mask_broadcast(
    shapes: List[Shape],
    mask_shape: Shape,
    scale: float,
    target: str,
    request,
    device,
):
    """
    Test Scaled Dot Product Attention Decode with different mask shapes.
    """
    batch = shapes[0][1]
    kv_seq = shapes[1][2]
    all_shapes = shapes + [mask_shape]
    dtypes = [
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        torch.bfloat16,
    ]

    def module(builder: TTIRBuilder):
        @builder.func(all_shapes, dtypes)
        def sdpa_decode_mask(
            query: Operand,
            key: Operand,
            value: Operand,
            cur_pos_tensor: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            result = builder.scaled_dot_product_attention_decode(
                query,
                key,
                value,
                cur_pos_tensor,
                attention_mask=attention_mask,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )

            cur_pos_data = torch.full((batch,), kv_seq - 1, dtype=torch.int32)
            builder.set_goldens({cur_pos_tensor: cur_pos_data})
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes,scale",
    [
        # --- No GQA (Q heads == KV heads) ---
        # Phi-1/1.5 decode: heads=32, kv_seq=128, head_dim=64
        (
            [(1, 32, 32, 64), (32, 32, 128, 64), (32, 32, 128, 64), (32,)],
            0.124999993,
        ),
        # Phi-2 decode: heads=32, kv_seq=128, head_dim=80 (not 32-aligned)
        (
            [(1, 32, 32, 80), (32, 32, 128, 80), (32, 32, 128, 80), (32,)],
            0.111803405,
        ),
        # --- GQA (Q heads != KV heads) ---
        # Llama 3.2 1B decode: heads=32/8 (4:1 GQA), kv_seq=128, head_dim=64
        (
            [(1, 32, 32, 64), (32, 8, 128, 64), (32, 8, 128, 64), (32,)],
            0.124999993,
        ),
        # Llama 3.1 8B decode: heads=32/8 (4:1 GQA), kv_seq=128, head_dim=128
        (
            [(1, 32, 32, 128), (32, 8, 128, 128), (32, 8, 128, 128), (32,)],
            0.0883883387,
        ),
        # Llama 3.2 3B decode: heads=24/8 (3:1 GQA), kv_seq=128, head_dim=128
        (
            [(1, 32, 24, 128), (32, 8, 128, 128), (32, 8, 128, 128), (32,)],
            0.0883883387,
        ),
        # Qwen 2.5 0.5B decode: heads=14/2 (7:1 GQA), kv_seq=128, head_dim=64
        (
            [(1, 32, 14, 64), (32, 2, 128, 64), (32, 2, 128, 64), (32,)],
            0.124999993,
        ),
        # Qwen 2.5 1.5B decode: heads=12/2 (6:1 GQA), kv_seq=128, head_dim=128
        (
            [(1, 32, 12, 128), (32, 2, 128, 128), (32, 2, 128, 128), (32,)],
            0.0883883387,
        ),
        # Qwen 2.5 3B decode: heads=16/2 (8:1 GQA), kv_seq=128, head_dim=128
        (
            [(1, 32, 16, 128), (32, 2, 128, 128), (32, 2, 128, 128), (32,)],
            0.0883883387,
        ),
        # Qwen 2.5 7B decode: heads=28/4 (7:1 GQA), kv_seq=128, head_dim=128
        (
            [(1, 32, 28, 128), (32, 4, 128, 128), (32, 4, 128, 128), (32,)],
            0.0883883387,
        ),
        # Qwen 3 0.6B/1.7B decode: heads=16/8 (2:1 GQA), kv_seq=128, head_dim=128
        (
            [(1, 32, 16, 128), (32, 8, 128, 128), (32, 8, 128, 128), (32,)],
            0.0883883387,
        ),
        # Qwen 3 4B/8B decode: heads=32/8 (4:1 GQA), kv_seq=128, head_dim=128
        (
            [(1, 32, 32, 128), (32, 8, 128, 128), (32, 8, 128, 128), (32,)],
            0.0883883387,
        ),
        # Falcon 3.1B decode: heads=8/4 (2:1 GQA), kv_seq=128, head_dim=256
        (
            [(1, 32, 8, 256), (32, 4, 128, 256), (32, 4, 128, 256), (32,)],
            6.250000,
        ),
        # Falcon 3.3B decode: heads=12/4 (3:1 GQA), kv_seq=128, head_dim=256
        pytest.param(
            [(1, 32, 12, 256), (32, 4, 128, 256), (32, 4, 128, 256), (32,)],
            6.250000,
            marks=pytest.mark.xfail(
                reason="PCC marginally below 0.99 threshold due to bf16 precision "
                "with large scale (6.25) and head_dim=256"
            ),
        ),
        # Gemma 1.1 2B decode: heads=8/1 (8:1 GQA), kv_seq=128, head_dim=256
        (
            [(1, 32, 8, 256), (32, 1, 128, 256), (32, 1, 128, 256), (32,)],
            6.250000,
        ),
        # Gemma 2 2B decode: heads=8/4 (2:1 GQA), kv_seq=128, head_dim=256
        (
            [(1, 32, 8, 256), (32, 4, 128, 256), (32, 4, 128, 256), (32,)],
            6.250000,
        ),
    ],
    ids=[
        "phi_1_decode",
        "phi_2_decode",
        "llama_3_2_1b_decode",
        "llama_3_1_8b_decode",
        "llama_3_2_3b_decode",
        "qwen_2_5_0_5b_decode",
        "qwen_2_5_1_5b_decode",
        "qwen_2_5_3b_decode",
        "qwen_2_5_7b_decode",
        "qwen_3_0_6b_decode",
        "qwen_3_4b_decode",
        "falcon_3_1b_decode",
        "falcon_3_3b_decode",
        "gemma_1_1_2b_decode",
        "gemma_2_2b_decode",
    ],
)
@pytest.mark.skip_exec(
    ("p150",),
    reason="SDPA decode kernel exceeds max runtime args on p150 (344 > 341). "
    "https://github.com/tenstorrent/tt-metal/issues/TBD",
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decode_in_models(
    shapes: List[Shape],
    scale: float,
    target: str,
    request,
    device,
):
    """
    Test Scaled Dot Product Attention Decode with real model shapes extracted
    from single_blocks_and_layers model tests after TTNNFusing.
    """
    batch = shapes[0][1]
    kv_seq = shapes[1][2]
    dtypes = [torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def sdpa_decode(
            query: Operand,
            key: Operand,
            value: Operand,
            cur_pos_tensor: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            result = builder.scaled_dot_product_attention_decode(
                query,
                key,
                value,
                cur_pos_tensor,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )

            # Override cur_pos_tensor with valid values (kv_seq - 1 so all
            # KV positions are attended to, matching the golden which attends
            # to all positions).
            cur_pos_data = torch.full((batch,), kv_seq - 1, dtype=torch.int32)
            builder.set_goldens({cur_pos_tensor: cur_pos_data})
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )
