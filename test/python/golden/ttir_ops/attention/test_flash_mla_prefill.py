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


# tt-metal's ttnn.transformer.flash_mla_prefill silently drops the attention_mask
# whenever use_mla=true
# Tracked upstream: https://github.com/tenstorrent/tt-metal/issues/43239
_MLA_MASK_UNSUPPORTED = pytest.mark.xfail(
    reason="tt-metal flash_mla_prefill ignores attn_mask when use_mla=true. "
    "Tracked upstream: tt-metal#43239.",
    strict=False,
)


# ---------------------------------------------------------------------------
# Causal MLA-from-latent (no value, no mask).
# Exercises operandSegmentSizes = [1, 1, 0, 0] and the head_dim_v <= qkHeadSize
# branch in the verifier.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shapes,head_dim_v,scale",
    [
        # Hq=Hkv=8 (vanilla MHA).
        ([(1, 8, 32, 128), (1, 8, 32, 128)], 64, 0.08838834764831845),
        # Hq=16, Hkv=1 (full MLA collapse).
        ([(1, 16, 32, 128), (1, 1, 32, 128)], 64, 0.08838834764831845),
        # 4:1 GQA, head_dim_v == qkHeadSize (boundary).
        ([(2, 8, 64, 128), (2, 2, 64, 128)], 128, 0.08838834764831845),
    ],
    ids=["mha", "mla_collapse", "gqa_head_dim_equal_qk"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_flash_mla_prefill_causal_no_value(
    shapes: List[Shape],
    head_dim_v: int,
    scale: float,
    target: str,
    request,
    device,
):
    dtypes = [torch.bfloat16] * len(shapes)

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def flash_mla_prefill_causal_no_value(
            query: Operand,
            key: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.flash_mla_prefill(
                query,
                key,
                head_dim_v=head_dim_v,
                is_causal=True,
                scale=scale,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


# ---------------------------------------------------------------------------
# Causal with explicit value.
# operandSegmentSizes = [1, 1, 1, 0]; exercises the value-shape verifier.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shapes,head_dim_v,scale",
    [
        # Hq=Hkv (MHA) with value.
        (
            [(1, 8, 32, 128), (1, 8, 32, 128), (1, 8, 32, 64)],
            64,
            0.08838834764831845,
        ),
        # GQA 4:1 with value.
        (
            [(2, 8, 64, 128), (2, 2, 64, 128), (2, 2, 64, 96)],
            96,
            0.08838834764831845,
        ),
    ],
    ids=["mha_with_value", "gqa_with_value"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_flash_mla_prefill_causal_with_value(
    shapes: List[Shape],
    head_dim_v: int,
    scale: float,
    target: str,
    request,
    device,
):
    dtypes = [torch.bfloat16] * len(shapes)

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def flash_mla_prefill_causal_with_value(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.flash_mla_prefill(
                query,
                key,
                head_dim_v=head_dim_v,
                value=value,
                is_causal=True,
                scale=scale,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


# ---------------------------------------------------------------------------
# Non-causal with attention mask (no value).
# Exercises the mutually-exclusive (mask, is_causal) verifier branch and the
# mask-shape broadcast permutations: [1|B x 1|Hq x Sq x Sq].
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shapes,mask_shape,head_dim_v,scale",
    [
        # Full mask broadcast (1, 1, Sq, Sq).
        (
            [(1, 16, 32, 128), (1, 1, 32, 128)],
            (1, 1, 32, 32),
            64,
            0.08838834764831845,
        ),
        # Per-batch mask (B, 1, Sq, Sq).
        (
            [(2, 16, 32, 128), (2, 1, 32, 128)],
            (2, 1, 32, 32),
            64,
            0.08838834764831845,
        ),
        # Per-head mask (1, Hq, Sq, Sq).
        (
            [(1, 16, 32, 128), (1, 1, 32, 128)],
            (1, 16, 32, 32),
            64,
            0.08838834764831845,
        ),
    ],
    ids=[
        "mask_broadcast_batch_and_heads",
        "mask_per_batch",
        "mask_per_head",
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
@_MLA_MASK_UNSUPPORTED
def test_flash_mla_prefill_with_mask(
    shapes: List[Shape],
    mask_shape: Shape,
    head_dim_v: int,
    scale: float,
    target: str,
    request,
    device,
):
    all_shapes = shapes + [mask_shape]
    dtypes = [torch.bfloat16] * len(all_shapes)

    def module(builder: TTIRBuilder):
        @builder.func(all_shapes, dtypes)
        def flash_mla_prefill_with_mask(
            query: Operand,
            key: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.flash_mla_prefill(
                query,
                key,
                head_dim_v=head_dim_v,
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


# ---------------------------------------------------------------------------
# All four operands present (query, key, value, mask) with explicit scale.
# operandSegmentSizes = [1, 1, 1, 1].
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shapes,head_dim_v,scale",
    [
        (
            [(2, 8, 64, 128), (2, 1, 64, 128), (2, 1, 64, 96), (2, 1, 64, 64)],
            96,
            0.125,
        ),
        (
            [(1, 16, 32, 128), (1, 1, 32, 128), (1, 1, 32, 64), (1, 1, 32, 32)],
            64,
            0.08838834764831845,
        ),
    ],
    ids=["mla_value_mask_scale_b2", "mla_value_mask_scale_b1"],
)
@pytest.mark.parametrize("target", ["ttnn"])
@_MLA_MASK_UNSUPPORTED
def test_flash_mla_prefill_value_mask_scale(
    shapes: List[Shape],
    head_dim_v: int,
    scale: float,
    target: str,
    request,
    device,
):
    dtypes = [torch.bfloat16] * len(shapes)

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def flash_mla_prefill_value_mask_scale(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.flash_mla_prefill(
                query,
                key,
                head_dim_v=head_dim_v,
                value=value,
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
