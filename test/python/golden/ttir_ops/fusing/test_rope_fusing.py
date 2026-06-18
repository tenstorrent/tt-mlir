# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from builder.base.builder_utils import Operand
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


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def torch_rope(x, cos, sin):
    """Golden reference for RoPE. Works for both patterns since they are
    mathematically equivalent. cos/sin are broadcast over batch and heads."""
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Pattern 1: rotate_half
#   result = x * cos + rotate_half(x) * sin
#   where rotate_half(x) = concat(neg(x[D/2:]), x[:D/2])
# ---------------------------------------------------------------------------


def build_rope_rotate_half(
    input: Operand,
    cos_input: Operand,
    sin_input: Operand,
    builder: TTIRBuilder,
):
    """Build Pattern 1 (rotate_half) as a sequence of TTIR ops.

    cos/sin are 3D [1, seq, head_dim] and get reshaped to 4D [1, 1, seq, head_dim]
    then broadcast to match the input [batch, heads, seq, head_dim].
    This mirrors how tt-xla emits the pattern for llama, falcon, qwen, etc.
    """
    cos_4d_shape = [1, 1] + cos_input.type.shape[1:]
    cos_reshaped = builder.reshape(cos_input, shape=cos_4d_shape)
    sin_reshaped = builder.reshape(sin_input, shape=cos_4d_shape)

    # x * cos (broadcast cos over batch and heads)
    x_cos = builder.multiply(input, cos_reshaped)

    last_dim = input.type.shape[-1]
    half_dim = last_dim // 2

    # rotate_half(x) = concat(neg(x[D/2:]), x[:D/2])
    begins_hi = [0, 0, 0, half_dim]
    ends_hi = list(input.type.shape[:3]) + [last_dim]
    x_hi = builder.slice(input, begins=begins_hi, ends=ends_hi, step=[1, 1, 1, 1])
    neg_hi = builder.neg(x_hi)

    begins_lo = [0, 0, 0, 0]
    ends_lo = list(input.type.shape[:3]) + [half_dim]
    x_lo = builder.slice(input, begins=begins_lo, ends=ends_lo, step=[1, 1, 1, 1])

    rotated = builder.concat([neg_hi, x_lo], dim=3)

    # rotate_half(x) * sin (broadcast sin over batch and heads)
    rot_sin = builder.multiply(rotated, sin_reshaped)

    return builder.add(x_cos, rot_sin)


# ---------------------------------------------------------------------------
# Pattern 2: complex rotation (expanded / trig-identity form)
#   real = x1*cos - x2*sin
#   imag = x2*cos + x1*sin
#   result = concat(real, imag)
#   where x1 = x[:D/2], x2 = x[D/2:]
# Used by gpt_oss models. cos/sin are half-dim and get doubled.
# ---------------------------------------------------------------------------


def build_rope_complex_rotation(
    input: Operand,
    cos_half_input: Operand,
    sin_half_input: Operand,
    builder: TTIRBuilder,
):
    """Build Pattern 2 (complex rotation) as a sequence of TTIR ops.

    cos/sin are half-dim 3D [1, seq, head_dim/2] and get reshaped to 4D
    then broadcast. This mirrors how tt-xla emits the gpt_oss pattern.
    """
    cos_4d_shape = [1, 1] + cos_half_input.type.shape[1:]
    cos_reshaped = builder.reshape(cos_half_input, shape=cos_4d_shape)
    sin_reshaped = builder.reshape(sin_half_input, shape=cos_4d_shape)

    last_dim = input.type.shape[-1]
    half_dim = last_dim // 2

    # x1 = x[:D/2], x2 = x[D/2:]
    begins_lo = [0, 0, 0, 0]
    ends_lo = list(input.type.shape[:3]) + [half_dim]
    x1 = builder.slice(input, begins=begins_lo, ends=ends_lo, step=[1, 1, 1, 1])

    begins_hi = [0, 0, 0, half_dim]
    ends_hi = list(input.type.shape[:3]) + [last_dim]
    x2 = builder.slice(input, begins=begins_hi, ends=ends_hi, step=[1, 1, 1, 1])

    # real = x1*cos - x2*sin
    x1_cos = builder.multiply(x1, cos_reshaped)
    x2_sin = builder.multiply(x2, sin_reshaped)
    real = builder.subtract(x1_cos, x2_sin)

    # imag = x2*cos + x1*sin
    x2_cos = builder.multiply(x2, cos_reshaped)
    x1_sin = builder.multiply(x1, sin_reshaped)
    imag = builder.add(x2_cos, x1_sin)

    return builder.concat([real, imag], dim=3)


# ---------------------------------------------------------------------------
# Test parameters: shapes extracted from 40 LLM TTIR graphs (tt-xla CI)
#
# Pattern 1 (rotate_half):
#   input: [B, H, S, D]  cos/sin: [1, S, D]  (3D, reshaped to 4D in builder)
#
# Pattern 2 (complex rotation):
#   input: [B, H, S, D]  cos/sin: [1, S, D/2] (half-dim, 3D)
# ---------------------------------------------------------------------------

# Representative shapes per model family for Pattern 1
ROTATE_HALF_SHAPES = [
    # --- Prefill (seq > 1) ---
    pytest.param(
        (32, 8, 18, 128),
        (1, 18, 128),
        id="llama_3_1_8b-prefill",
    ),
    pytest.param(
        (32, 8, 18, 64),
        (1, 18, 64),
        id="llama_3_2_1b-prefill",
    ),
    pytest.param(
        (32, 4, 17, 256),
        (1, 17, 256),
        id="falcon3_1b-prefill",
    ),
    pytest.param(
        (32, 32, 17, 32),
        (1, 17, 32),
        id="phi1_5-prefill",
    ),
    pytest.param(
        (32, 2, 17, 128),
        (1, 17, 128),
        id="qwen_2_5_1_5b-prefill",
    ),
    pytest.param(
        (32, 8, 18, 128),
        (1, 18, 128),
        id="mistral_7b-prefill",
    ),
    pytest.param(
        (32, 1, 17, 256),
        (1, 17, 256),
        id="gemma_1_1_2b-prefill",
    ),
    # --- Decode (seq = 1) ---
    pytest.param(
        (32, 8, 1, 128),
        (1, 1, 128),
        id="llama_3_1_8b-decode",
    ),
    pytest.param(
        (32, 8, 1, 64),
        (1, 1, 64),
        id="llama_3_2_1b-decode",
    ),
    pytest.param(
        (32, 4, 1, 256),
        (1, 1, 256),
        id="falcon3_1b-decode",
    ),
    pytest.param(
        (32, 32, 1, 32),
        (1, 1, 32),
        id="phi1_5-decode",
    ),
    pytest.param(
        (32, 2, 1, 128),
        (1, 1, 128),
        id="qwen_2_5_1_5b-decode",
    ),
    pytest.param(
        (32, 8, 1, 128),
        (1, 1, 128),
        id="ministral_8b-decode",
    ),
]

# Representative shapes for Pattern 2 (complex rotation / expanded)
COMPLEX_ROTATION_SHAPES = [
    pytest.param(
        (32, 8, 17, 64),
        (1, 17, 32),
        id="gpt_oss_20b_tp-prefill",
    ),
    pytest.param(
        (32, 8, 1, 64),
        (1, 1, 32),
        id="gpt_oss_20b_tp-decode",
    ),
    pytest.param(
        (64, 8, 17, 64),
        (1, 17, 32),
        id="gpt_oss_120b_tp_galaxy-prefill",
    ),
    pytest.param(
        (1, 8, 17, 64),
        (1, 17, 32),
        id="gpt_oss_20b_tp_batch_size_1-prefill",
    ),
]


@pytest.mark.parametrize("input_shape, cos_sin_shape", ROTATE_HALF_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rope_rotate_half(input_shape, cos_sin_shape, dtype, target, request, device):
    """
    Pattern 1: rotate_half RoPE.
    Used by llama, falcon, gemma, qwen, mistral, ministral, phi, kimi.

    Shapes extracted from tt-xla CI artifacts (40 LLM models).
    Verifies that the decomposed RoPE pattern fuses into ttnn.rotary_embedding
    through the full pipeline.
    """
    shapes = [input_shape, cos_sin_shape, cos_sin_shape]
    dtypes = [dtype] * 3

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rotary_embedding(
            input: Operand,
            cos_input: Operand,
            sin_input: Operand,
            builder: TTIRBuilder,
        ):
            input_data = torch.randn(input_shape, dtype=dtype)
            cos_data = torch.randn(cos_sin_shape, dtype=dtype)
            sin_data = torch.randn(cos_sin_shape, dtype=dtype)

            cos_4d = cos_data.unsqueeze(0)
            sin_4d = sin_data.unsqueeze(0)
            golden = torch_rope(input_data, cos_4d, sin_4d)

            result = build_rope_rotate_half(input, cos_input, sin_input, builder)

            builder.set_goldens(
                {input: input_data, cos_input: cos_data, sin_input: sin_data},
                {result: golden},
            )
            return result

    output = compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=[
            "enable-ttnn-decomposition-pass=false",
            "composite-resolution=force-promote",
        ],
        save_artifacts=True,
    )

    assert check_op(output, "rotary_embedding")


@pytest.mark.parametrize("input_shape, cos_sin_half_shape", COMPLEX_ROTATION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rope_complex_rotation(
    input_shape, cos_sin_half_shape, dtype, target, request, device
):
    """
    Pattern 2: complex rotation (expanded / trig-identity) RoPE.
    Used by gpt_oss_20b and gpt_oss_120b.

    cos/sin are half-dim [1, S, D/2]. The fusion doubles them to full-dim
    by concatenating with themselves before creating ttnn.rotary_embedding.

    Previously unfused at TTNN level when cos/sin were pre-scaled (#8415).
    Now fused at TTIR level where the scaled values are absorbed naturally.
    """
    shapes = [input_shape, cos_sin_half_shape, cos_sin_half_shape]
    dtypes = [dtype] * 3

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rotary_embedding(
            input: Operand,
            cos_half_input: Operand,
            sin_half_input: Operand,
            builder: TTIRBuilder,
        ):
            input_data = torch.randn(input_shape, dtype=dtype)
            cos_half_data = torch.randn(cos_sin_half_shape, dtype=dtype)
            sin_half_data = torch.randn(cos_sin_half_shape, dtype=dtype)

            cos_full = torch.cat([cos_half_data, cos_half_data], dim=-1)
            sin_full = torch.cat([sin_half_data, sin_half_data], dim=-1)
            golden = torch_rope(
                input_data, cos_full.unsqueeze(0), sin_full.unsqueeze(0)
            )

            result = build_rope_complex_rotation(
                input, cos_half_input, sin_half_input, builder
            )

            builder.set_goldens(
                {
                    input: input_data,
                    cos_half_input: cos_half_data,
                    sin_half_input: sin_half_data,
                },
                {result: golden},
            )
            return result

    output = compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=[
            "enable-ttnn-decomposition-pass=false",
            "composite-resolution=force-promote",
        ],
        save_artifacts=True,
    )

    assert check_op(output, "rotary_embedding")


# ---------------------------------------------------------------------------
# Decomposition quality: verify the composite decomposition produces the
# complex rotation form (half-D ops, no neg) when inlined, matching PR #8580.
#
# Uses Qwen3-0.6B shapes from tt-xla#4886 — decode (1,4,1,128) followed by
# prefill (1,4,64,128) on the same device. With the rotate_half fallback,
# decode emits concat(1,1,1,64) which collides in the tt-metal program cache
# with prefill's rotate_half concat(1,4,64,64) (tt-metal#45089). The complex
# rotation form avoids this by never emitting the (1,1,1,D/2) concat.
# ---------------------------------------------------------------------------


def _build_and_run_complex_rotation(
    input_shape, cos_sin_half_shape, dtype, target, request, device
):
    """Helper: build complex rotation RoPE, compile, and execute."""
    shapes = [input_shape, cos_sin_half_shape, cos_sin_half_shape]
    dtypes = [dtype] * 3

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rotary_embedding(
            input: Operand,
            cos_half_input: Operand,
            sin_half_input: Operand,
            builder: TTIRBuilder,
        ):
            input_data = torch.randn(input_shape, dtype=dtype)
            cos_half_data = torch.randn(cos_sin_half_shape, dtype=dtype)
            sin_half_data = torch.randn(cos_sin_half_shape, dtype=dtype)

            cos_full = torch.cat([cos_half_data, cos_half_data], dim=-1)
            sin_full = torch.cat([sin_half_data, sin_half_data], dim=-1)
            golden = torch_rope(
                input_data, cos_full.unsqueeze(0), sin_full.unsqueeze(0)
            )

            result = build_rope_complex_rotation(
                input, cos_half_input, sin_half_input, builder
            )

            builder.set_goldens(
                {
                    input: input_data,
                    cos_half_input: cos_half_data,
                    sin_half_input: sin_half_data,
                },
                {result: golden},
            )
            return result

    return compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rope_complex_rotation_decomposition(dtype, target, request, device):
    """
    Run decode then prefill on the same device (shared program cache) using
    Qwen3-0.6B shapes from tt-xla#4886. Verifies:
    1. The inlined decomposition uses complex rotation form (subtract, no neg).
    2. Both shapes execute without tt-metal#45089 program-cache collision.
    """
    # Decode: seq=1, 4 KV heads, head_dim=128, cos/sin half-dim=64
    decode_output = _build_and_run_complex_rotation(
        (1, 4, 1, 128), (1, 1, 64), dtype, target, request, device
    )
    # Prefill: seq=64
    prefill_output = _build_and_run_complex_rotation(
        (1, 4, 64, 128), (1, 64, 64), dtype, target, request, device
    )

    # Both should be inlined (not promoted) — no ttnn.rotary_embedding.
    assert not check_op(decode_output, "rotary_embedding")
    assert not check_op(prefill_output, "rotary_embedding")
    # Complex rotation form uses subtract, not neg.
    assert check_op(decode_output, "subtract")
    assert not check_op(decode_output, "neg")
    assert check_op(prefill_output, "subtract")
    assert not check_op(prefill_output, "neg")


# ---------------------------------------------------------------------------
# Pattern 3: interleaved-pair
#   x_   = reshape(x, [..., D/2, 1, 2])
#   out  = reshape(freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1],
#                  [..., D])
#   where freqs is shape (..., D/2, 2, 2) packing per-pair [[cos,-sin],[sin,cos]]
# ---------------------------------------------------------------------------


def torch_rope_interleaved_pair(x, freqs):
    """Golden reference for interleaved-pair RoPE."""
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


def build_rope_interleaved_pair(
    input: Operand,
    freqs_input: Operand,
    builder: TTIRBuilder,
):
    """Build Pattern 3 (interleaved-pair) as a sequence of TTIR ops.

    input is [B, H, S, D]; freqs is [B, 1, S, D/2, 2, 2] with heads=1 broadcast
    across H.
    """
    input_shape = list(input.type.shape)
    freqs_shape = list(freqs_input.type.shape)
    B, H, S, D = input_shape
    half_dim = D // 2

    # x_ = reshape(x, [..., D/2, 1, 2])
    x_6d = builder.reshape(input, shape=[B, H, S, half_dim, 1, 2])

    # x_[..., 0] (real) and x_[..., 1] (imag) — slice + reshape + broadcast
    # to align with the (..., D/2, 2) shape used by the multiplies.
    x_p0 = builder.slice(
        x_6d,
        begins=[0, 0, 0, 0, 0, 0],
        ends=[B, H, S, half_dim, 1, 1],
        step=[1, 1, 1, 1, 1, 1],
    )
    x_p1 = builder.slice(
        x_6d,
        begins=[0, 0, 0, 0, 0, 1],
        ends=[B, H, S, half_dim, 1, 2],
        step=[1, 1, 1, 1, 1, 1],
    )
    x_real_bc = builder.broadcast(
        builder.reshape(
            builder.reshape(x_p0, shape=[B, H, S, half_dim]),
            shape=[B, H, S, half_dim, 1],
        ),
        broadcast_dimensions=[1, 1, 1, 1, 2],
    )
    x_imag_bc = builder.broadcast(
        builder.reshape(
            builder.reshape(x_p1, shape=[B, H, S, half_dim]),
            shape=[B, H, S, half_dim, 1],
        ),
        broadcast_dimensions=[1, 1, 1, 1, 2],
    )

    # freqs[..., 0] = [cos, sin] and freqs[..., 1] = [-sin, cos]
    # — slice the last dim, then broadcast over the heads dim.
    freqs_c0 = builder.slice(
        freqs_input,
        begins=[0, 0, 0, 0, 0, 0],
        ends=[freqs_shape[0], freqs_shape[1], freqs_shape[2], half_dim, 2, 1],
        step=[1, 1, 1, 1, 1, 1],
    )
    freqs_c1 = builder.slice(
        freqs_input,
        begins=[0, 0, 0, 0, 0, 1],
        ends=[freqs_shape[0], freqs_shape[1], freqs_shape[2], half_dim, 2, 2],
        step=[1, 1, 1, 1, 1, 1],
    )
    # Squeeze the size-1 heads-broadcast dim (position [1]) and the trailing
    # size-1 dim from the column slice, then re-add the heads-broadcast dim.
    cos_bc = builder.broadcast(
        builder.reshape(
            builder.reshape(
                freqs_c0, shape=[freqs_shape[0], freqs_shape[2], half_dim, 2]
            ),
            shape=[freqs_shape[0], 1, freqs_shape[2], half_dim, 2],
        ),
        broadcast_dimensions=[1, H, 1, 1, 1],
    )
    sin_bc = builder.broadcast(
        builder.reshape(
            builder.reshape(
                freqs_c1, shape=[freqs_shape[0], freqs_shape[2], half_dim, 2]
            ),
            shape=[freqs_shape[0], 1, freqs_shape[2], half_dim, 2],
        ),
        broadcast_dimensions=[1, H, 1, 1, 1],
    )

    # freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1], reshaped to [..., D]
    sum_5d = builder.add(
        builder.multiply(cos_bc, x_real_bc),
        builder.multiply(sin_bc, x_imag_bc),
    )
    return builder.reshape(sum_5d, shape=input_shape)


# Representative shapes for Pattern 3 (interleaved-pair)
INTERLEAVED_PAIR_SHAPES = [
    pytest.param(
        (1, 20, 128, 128),
        (1, 1, 128, 64, 2, 2),
        id="hidream_i1_fast-prefill",
    ),
]


@pytest.mark.parametrize("input_shape, freqs_shape", INTERLEAVED_PAIR_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rope_interleaved_pair(
    input_shape, freqs_shape, dtype, target, request, device
):
    """
    Pattern 3: interleaved-pair RoPE.
    Used by HiDream-I1.

    Verifies that the decomposed RoPE pattern fuses into ttnn.rotary_embedding
    through the full pipeline.
    """
    shapes = [input_shape, freqs_shape]
    dtypes = [dtype] * 2

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rotary_embedding(
            input: Operand,
            freqs_input: Operand,
            builder: TTIRBuilder,
        ):
            input_data = torch.randn(input_shape, dtype=dtype)

            # Build a valid freqs_cis: per pair, 2x2 = [[c, -s], [s, c]].
            angles = torch.randn(freqs_shape[:-2], dtype=torch.float32)
            c, s = torch.cos(angles), torch.sin(angles)
            freqs_data = (
                torch.stack([c, -s, s, c], dim=-1).reshape(freqs_shape).to(dtype)
            )

            golden = torch_rope_interleaved_pair(input_data, freqs_data)

            result = build_rope_interleaved_pair(input, freqs_input, builder)

            builder.set_goldens(
                {input: input_data, freqs_input: freqs_data},
                {result: golden},
            )
            return result

    output = compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=[
            "enable-ttnn-decomposition-pass=false",
            "composite-resolution=force-promote",
        ],
        save_artifacts=True,
    )

    assert check_op(output, "rotary_embedding")
