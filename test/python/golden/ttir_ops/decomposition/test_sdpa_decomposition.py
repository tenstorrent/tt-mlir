# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Optional
from dataclasses import dataclass

import pytest
import torch

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs, DeferredDevice

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
    def scale(self):
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def is_gqa(self):
        return self.q_heads != self.kv_heads

    # SDPADecode shape: Q is [1, batch, heads, head_dim]
    @property
    def decode_query(self):
        return (1, self.batch, self.q_heads, self.head_dim)

    @property
    def decode_mask(self):
        return (self.batch, 1, 1, self.kv_seq)


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

GQA_SHAPES = [s for s in ALL_SHAPES if s.values[0].is_gqa]


# ---------------------------------------------------------------------------
# Golden computation
# ---------------------------------------------------------------------------


def build_torch_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    enable_gqa = q_heads != kv_heads
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        scale=scale,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_op(mlir_file: str, op_name: str) -> bool:
    op_name = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if op_name in line:
                return True
    return False


def assert_sdpa_decomposed(mlir_path: str):
    """Verify that no fused SDPA ops remain in the compiled MLIR."""
    assert not check_op(
        mlir_path, "scaled_dot_product_attention_decode"
    ), "scaled_dot_product_attention_decode should have been decomposed"
    assert not check_op(
        mlir_path, "scaled_dot_product_attention"
    ), "scaled_dot_product_attention should have been decomposed"


def compile_and_run_decompose(module_fn, target, request):
    return compile_and_execute_ttir(
        module_fn,
        target=target,
        **get_request_kwargs(request),
        device=DeferredDevice(request),
        pipeline_options=["enable-optimizer=true", "ttnn-force-decompose=true"],
        save_artifacts=True,
    )


# ---------------------------------------------------------------------------
# Tests: SDPA decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sdpa_shapes", ALL_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decompose_with_mask(sdpa_shapes, target, request):
    """SDPA with explicit attention mask, force-decomposed to component ops."""
    input_shapes = [
        sdpa_shapes.query,
        sdpa_shapes.key,
        sdpa_shapes.value,
        sdpa_shapes.mask,
    ]
    dtypes = [torch.bfloat16] * 4
    scale = sdpa_shapes.scale

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(
            query: Operand,
            key: Operand,
            value: Operand,
            mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            m_data = torch.randn(sdpa_shapes.mask, dtype=torch.bfloat16)
            golden = build_torch_golden(
                q_data, k_data, v_data, scale=scale, attention_mask=m_data
            )
            result = builder.scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask=mask,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )
            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data, mask: m_data},
                {result: golden},
            )
            return result

    mlir_path = compile_and_run_decompose(module, target, request)
    assert_sdpa_decomposed(mlir_path)


@pytest.mark.parametrize("sdpa_shapes", ALL_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decompose_no_mask(sdpa_shapes, target, request):
    """SDPA without mask and is_causal=false, force-decomposed."""
    input_shapes = [sdpa_shapes.query, sdpa_shapes.key, sdpa_shapes.value]
    dtypes = [torch.bfloat16] * 3
    scale = sdpa_shapes.scale

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            golden = build_torch_golden(q_data, k_data, v_data, scale=scale)
            result = builder.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )
            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data},
                {result: golden},
            )
            return result

    mlir_path = compile_and_run_decompose(module, target, request)
    assert_sdpa_decomposed(mlir_path)


@pytest.mark.parametrize("sdpa_shapes", PREFILL_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decompose_causal(sdpa_shapes, target, request):
    """SDPA with is_causal=true, no explicit mask, force-decomposed."""
    input_shapes = [sdpa_shapes.query, sdpa_shapes.key, sdpa_shapes.value]
    dtypes = [torch.bfloat16] * 3
    scale = sdpa_shapes.scale

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            q_data = torch.randn(sdpa_shapes.query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            golden = build_torch_golden(
                q_data, k_data, v_data, scale=scale, is_causal=True
            )
            result = builder.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=scale,
                unit_attrs=unit_attrs,
            )
            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data},
                {result: golden},
            )
            return result

    mlir_path = compile_and_run_decompose(module, target, request)
    assert_sdpa_decomposed(mlir_path)


# ---------------------------------------------------------------------------
# Tests: SDPADecode decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sdpa_shapes", DECODE_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decode_decompose_with_mask(sdpa_shapes, target, request):
    """SDPADecode with mask, force-decomposed to permute -> SDPA component ops -> permute."""
    input_shapes = [
        sdpa_shapes.decode_query,
        sdpa_shapes.key,
        sdpa_shapes.value,
        sdpa_shapes.decode_mask,
    ]
    dtypes = [torch.bfloat16] * 4
    scale = sdpa_shapes.scale

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_decode(
            query: Operand,
            key: Operand,
            value: Operand,
            mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # SDPADecode Q: [1, B, H, D] -> permute to [B, H, 1, D] for golden
            q_data = torch.randn(sdpa_shapes.decode_query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            m_data = torch.randn(sdpa_shapes.decode_mask, dtype=torch.bfloat16)
            # Permute Q to standard SDPA shape for golden computation
            q_permuted = q_data.permute(1, 2, 0, 3)  # [B, H, 1, D]
            golden_sdpa = build_torch_golden(
                q_permuted, k_data, v_data, scale=scale, attention_mask=m_data
            )
            # Permute golden back to decode shape [1, B, H, D]
            golden = golden_sdpa.permute(2, 0, 1, 3)
            result = builder.scaled_dot_product_attention_decode(
                query,
                key,
                value,
                attention_mask=mask,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )
            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data, mask: m_data},
                {result: golden},
            )
            return result

    mlir_path = compile_and_run_decompose(module, target, request)
    assert_sdpa_decomposed(mlir_path)


@pytest.mark.parametrize("sdpa_shapes", DECODE_SHAPES)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decode_decompose_no_mask(sdpa_shapes, target, request):
    """SDPADecode without mask, force-decomposed."""
    input_shapes = [
        sdpa_shapes.decode_query,
        sdpa_shapes.key,
        sdpa_shapes.value,
    ]
    dtypes = [torch.bfloat16] * 3
    scale = sdpa_shapes.scale

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def sdpa_decode(
            query: Operand,
            key: Operand,
            value: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            q_data = torch.randn(sdpa_shapes.decode_query, dtype=torch.bfloat16)
            k_data = torch.randn(sdpa_shapes.key, dtype=torch.bfloat16)
            v_data = torch.randn(sdpa_shapes.value, dtype=torch.bfloat16)
            q_permuted = q_data.permute(1, 2, 0, 3)
            golden_sdpa = build_torch_golden(q_permuted, k_data, v_data, scale=scale)
            golden = golden_sdpa.permute(2, 0, 1, 3)
            result = builder.scaled_dot_product_attention_decode(
                query,
                key,
                value,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )
            builder.set_goldens(
                {query: q_data, key: k_data, value: v_data},
                {result: golden},
            )
            return result

    mlir_path = compile_and_run_decompose(module, target, request)
    assert_sdpa_decomposed(mlir_path)
