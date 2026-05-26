# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import pytest
import torch

from builder.base.builder_utils import Operand, Shape, DeferredDevice
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compile_decomposed(module_fn, target, request):
    # enable-optimizer=false routes through TTNNPipelines.cpp:349, which adds
    # bare createTTNNDecomposition() with no validation config — SDPA gets
    # decomposed unconditionally regardless of shape.
    return compile_and_execute_ttir(
        module_fn,
        target=target,
        **get_request_kwargs(request),
        device=DeferredDevice(request),
        pipeline_options=["enable-optimizer=false"],
        save_artifacts=True,
    )


def check_op(mlir_file: str, op_name: str) -> bool:
    qualified = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if qualified in line:
                return True
    return False


def assert_decomposed(mlir_path: str):
    # SDPA ops fully decomposed away.
    assert not check_op(
        mlir_path, "scaled_dot_product_attention"
    ), "ttnn.scaled_dot_product_attention should not be present after decomposition"
    assert not check_op(
        mlir_path, "scaled_dot_product_attention_decode"
    ), "ttnn.scaled_dot_product_attention_decode should not be present after decomposition"
    # Primitive ops produced by the rewrite present.
    assert check_op(mlir_path, "matmul"), "expected ttnn.matmul in decomposed IR"
    assert check_op(mlir_path, "softmax"), "expected ttnn.softmax in decomposed IR"


# ---------------------------------------------------------------------------
# Prefill / non-decode tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shapes,mask_shape,is_causal,scale",
    [
        # MHA + explicit mask (mirrors sdpa_mha_with_mask in the lit test).
        (
            [(1, 8, 64, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
            (1, 1, 64, 64),
            False,
            0.125,
        ),
        # GQA, 4:1 (mirrors sdpa_gqa in the lit test).
        (
            [(1, 32, 64, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
            (1, 1, 64, 64),
            False,
            0.125,
        ),
    ],
    ids=["mha_with_mask", "gqa"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decomposition_with_mask(
    shapes: List[Shape],
    mask_shape: Shape,
    is_causal: bool,
    scale: float,
    target: str,
    request,
):
    all_shapes = shapes + [mask_shape]
    dtypes = [torch.bfloat16] * len(all_shapes)

    def module(builder: TTIRBuilder):
        @builder.func(all_shapes, dtypes)
        def sdpa(
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
                is_causal=is_causal,
                scale=scale,
                unit_attrs=unit_attrs,
            )

    mlir_path = compile_decomposed(module, target, request)
    assert_decomposed(mlir_path)


@pytest.mark.parametrize(
    "shapes,scale",
    [
        # Causal SDPA without an explicit mask (mirrors sdpa_causal_no_mask
        # in the lit test). is_causal=True forces the pattern to synthesize
        # a causal mask via the constant + add path.
        (
            [(1, 8, 64, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
            0.125,
        ),
    ],
    ids=["causal_no_mask"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decomposition_causal_no_mask(
    shapes: List[Shape],
    scale: float,
    target: str,
    request,
):
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
                is_causal=True,
                scale=scale,
                unit_attrs=unit_attrs,
            )

    mlir_path = compile_decomposed(module, target, request)
    assert_decomposed(mlir_path)


# ---------------------------------------------------------------------------
# Decode tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shapes,mask_shape,scale",
    [
        # Decode MHA (mirrors sdpa_decode_mha in the lit test).
        (
            [(1, 32, 32, 64), (32, 32, 128, 64), (32, 32, 128, 64), (32,)],
            (32, 1, 1, 128),
            0.125,
        ),
    ],
    ids=["decode_mha"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decode_decomposition_with_mask(
    shapes: List[Shape],
    mask_shape: Shape,
    scale: float,
    target: str,
    request,
):
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
        def sdpa_decode(
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

    mlir_path = compile_decomposed(module, target, request)
    assert_decomposed(mlir_path)


@pytest.mark.parametrize(
    "shapes,scale",
    [
        # Decode causal, no mask (mirrors sdpa_decode_causal in the lit test).
        (
            [(1, 32, 32, 64), (32, 32, 128, 64), (32, 32, 128, 64), (32,)],
            0.125,
        ),
    ],
    ids=["decode_causal"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decode_decomposition_causal_no_mask(
    shapes: List[Shape],
    scale: float,
    target: str,
    request,
):
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
            # Use a non-terminal cur_pos so the synthesized causal mask in the
            # decomposition actually masks future positions. cur_pos = kv_seq-1
            # would make the mask all-zeros and hide bugs in cur_pos handling.
            # NOTE: set_goldens must precede the SDPA call so the golden uses
            # the override (the SDPA's golden is computed at builder time).
            cur_pos_data = torch.full((batch,), kv_seq // 2, dtype=torch.int32)
            builder.set_goldens({cur_pos_tensor: cur_pos_data})
            result = builder.scaled_dot_product_attention_decode(
                query,
                key,
                value,
                cur_pos_tensor,
                is_causal=True,
                scale=scale,
                unit_attrs=unit_attrs,
            )
            return result

    mlir_path = compile_decomposed(module, target, request)
    assert_decomposed(mlir_path)


# ---------------------------------------------------------------------------
# Attention sink and sliding window (prefill only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shapes,mask_shape,sink_shape,scale",
    [
        # Prefill MHA + attention_sink (mirrors sdpa_with_attention_sink in
        # the lit test). Sink shape [1, Hq, 1, 1] per op definition.
        (
            [(1, 8, 64, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
            (1, 1, 64, 64),
            (1, 8, 1, 1),
            0.125,
        ),
    ],
    ids=["mha_with_sink"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decomposition_with_attention_sink(
    shapes: List[Shape],
    mask_shape: Shape,
    sink_shape: Shape,
    scale: float,
    target: str,
    request,
):
    all_shapes = shapes + [mask_shape, sink_shape]
    dtypes = [torch.bfloat16] * len(all_shapes)

    def module(builder: TTIRBuilder):
        @builder.func(all_shapes, dtypes)
        def sdpa(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            attention_sink: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                attention_sink=attention_sink,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )

    mlir_path = compile_decomposed(module, target, request)
    assert_decomposed(mlir_path)
    # Sink decomposition emits a concat (sink columns) then a slice afterwards.
    assert check_op(mlir_path, "concat"), "expected ttnn.concat for sink path"
    assert check_op(
        mlir_path, "slice_static"
    ), "expected ttnn.slice_static for sink path"


@pytest.mark.parametrize(
    "shapes,is_causal,sliding_window_size,scale",
    [
        # Prefill causal sliding window (mirrors sdpa_sliding_window in the
        # lit test). Window covers last W=32 tokens.
        (
            [(1, 8, 64, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
            True,
            32,
            0.125,
        ),
        # Prefill non-causal (bidirectional) sliding window. Window of size
        # W=32 centered on each query position. Exercises the non-causal
        # branch of generateSlidingWindowMask, which previously had a width
        # bug (used 2W-1 instead of W+1).
        (
            [(1, 8, 64, 64), (1, 8, 64, 64), (1, 8, 64, 64)],
            False,
            32,
            0.125,
        ),
    ],
    ids=["causal_sliding_window", "noncausal_sliding_window"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_decomposition_sliding_window(
    shapes: List[Shape],
    is_causal: bool,
    sliding_window_size: int,
    scale: float,
    target: str,
    request,
):
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
                is_causal=is_causal,
                scale=scale,
                sliding_window_size=sliding_window_size,
                unit_attrs=unit_attrs,
            )

    mlir_path = compile_decomposed(module, target, request)
    assert_decomposed(mlir_path)
    # Window mask is emitted as a ttnn.constant + add.
    assert check_op(mlir_path, "constant"), "expected ttnn.constant for window mask"
