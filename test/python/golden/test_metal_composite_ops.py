# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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

# =============================================================================
# SDPA decomposition tests — exercise is_causal, scale, attention_mask
# =============================================================================


@pytest.mark.parametrize(
    "shapes,is_causal,scale",
    [
        # MHA causal (Sq == Sk)
        (
            [(1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)],
            True,
            None,
        ),
        # MHA non-causal
        (
            [(1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)],
            False,
            None,
        ),
        # MHA causal with explicit scale
        (
            [(1, 8, 64, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
            True,
            0.25,
        ),
        # Llama 3.2 3B prefill: GQA 3:1, is_causal=true, Sq=128
        (
            [(1, 24, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128)],
            True,
            None,
        ),
        # Llama 3.2 3B GQA 3:1, is_causal=true, Sq=32
        (
            [(1, 24, 32, 128), (1, 8, 32, 128), (1, 8, 32, 128)],
            True,
            None,
        ),
        # Llama 3.2 3B GQA 3:1, is_causal=true, Sq=64
        (
            [(1, 24, 64, 128), (1, 8, 64, 128), (1, 8, 64, 128)],
            True,
            None,
        ),
        # GQA non-causal
        (
            [(1, 24, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128)],
            False,
            None,
        ),
        # GQA causal with custom scale
        (
            [(1, 24, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128)],
            True,
            0.1,
        ),
        # Qwen 4:1 GQA with mask (from sdpa.mlir), non-causal
        (
            [(1, 32, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128)],
            False,
            None,
        ),
    ],
    ids=[
        "mha_causal",
        "mha_non_causal",
        "mha_causal_custom_scale",
        "llama_3b_gqa_causal_sq128",
        "llama_3b_gqa_causal_sq32",
        "llama_3b_gqa_causal_sq64",
        "gqa_non_causal",
        "gqa_causal_custom_scale",
        "qwen_gqa_non_causal",
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_sdpa_decomposition(
    shapes: List[Shape],
    is_causal: bool,
    scale: Optional[float],
    target: str,
    request,
    device,
):
    """Test SDPA decomposition for the TTMetal pipeline with various attributes."""
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
            kwargs = dict(is_causal=is_causal, unit_attrs=unit_attrs)
            if scale is not None:
                kwargs["scale"] = scale
            return builder.scaled_dot_product_attention(query, key, value, **kwargs)

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes,mask_shape,is_causal,scale",
    [
        # Explicit mask, non-causal (broadcast batch+heads)
        (
            [(1, 8, 64, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
            (1, 1, 64, 64),
            False,
            0.125,
        ),
        # Qwen GQA 4:1 with mask (from sdpa.mlir)
        (
            [(1, 32, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128)],
            (1, 1, 128, 128),
            False,
            None,
        ),
    ],
    ids=["mask_broadcast", "qwen_gqa_mask"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_sdpa_decomposition_with_mask(
    shapes: List[Shape],
    mask_shape: Shape,
    is_causal: bool,
    scale: Optional[float],
    target: str,
    request,
    device,
):
    """Test SDPA decomposition with explicit attention mask."""
    all_shapes = shapes + [mask_shape]
    dtypes = [torch.bfloat16] * len(all_shapes)

    def module(builder: TTIRBuilder):
        @builder.func(all_shapes, dtypes)
        def sdpa_masked(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            kwargs = dict(
                attention_mask=attention_mask,
                is_causal=is_causal,
                unit_attrs=unit_attrs,
            )
            if scale is not None:
                kwargs["scale"] = scale
            return builder.scaled_dot_product_attention(query, key, value, **kwargs)

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


# =============================================================================
# RMS norm decomposition tests — exercise weight, bias, epsilon
# =============================================================================


@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((1, 32, 128), [128]),
        ((2, 4, 64), [64]),
    ],
)
@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_rms_norm_decomposition(
    shape: Shape,
    normalized_shape: List[int],
    has_weight: bool,
    has_bias: bool,
    epsilon: float,
    target: str,
    request,
    device,
):
    """Test RMS norm decomposition for the TTMetal pipeline."""
    shapes = [shape]
    if has_weight:
        shapes.append(tuple(normalized_shape))
    if has_bias:
        shapes.append(tuple(normalized_shape))

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def rms_norm(*inputs, unit_attrs: Optional[List[str]] = None):
            builder = inputs[-1]
            in0 = inputs[0]
            weight = None
            bias = None
            input_idx = 1

            if has_weight:
                weight = inputs[input_idx]
                input_idx += 1
            if has_bias:
                bias = inputs[input_idx]

            return builder.rms_norm(
                in0,
                normalized_shape=normalized_shape,
                weight=weight,
                bias=bias,
                epsilon=epsilon,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


# =============================================================================
# Softmax decomposition tests — exercise dimension
# =============================================================================


@pytest.mark.parametrize("shape", [(4, 64, 128)])
@pytest.mark.parametrize("dimension", [0, 1, 2, -1])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_softmax_decomposition(
    shape: Shape,
    dimension: int,
    target: str,
    request,
    device,
):
    """Test softmax decomposition for the TTMetal pipeline."""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def softmax(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.softmax(
                in0,
                dimension=dimension,
                numeric_stable=False,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )
