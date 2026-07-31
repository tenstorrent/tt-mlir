# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Composed full-shape Llama prefill decoder-block validation."""

import math
from pathlib import Path

import torch

from runner import InputSpec, PatternTest

_ROOT = Path(__file__).resolve().parents[4]
_MODEL = (
    _ROOT
    / "test"
    / "ttmlir"
    / "models"
    / "single_blocks_and_layers"
    / "llama_3_8b_prefill_decoder_block.mlir"
)
_MODEL_TEXT = _MODEL.read_text()
_WEIGHT_GROUPS = 8


def _uniform(shape, dtype, generator, radius):
    values = torch.rand(shape, dtype=dtype, generator=generator)
    return values.mul_(2.0 * radius).sub_(radius)


def _structured_weight(shape, dtype, generator):
    rows, columns = shape
    base = 1.0 / (128.0 if shape == (4096, 14336) else 64.0)
    choices = torch.tensor([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=dtype)
    row_indices = torch.randint(
        choices.numel(),
        (rows, _WEIGHT_GROUPS),
        generator=generator,
    )
    row_factors = choices[row_indices]
    row_factors[0, :] = 1.0

    column_indices = torch.randint(
        choices.numel(),
        (columns,),
        generator=generator,
    )
    column_factors = choices[column_indices].mul_(base)
    group_indices = torch.arange(columns).mul(_WEIGHT_GROUPS).floor_divide(columns)
    return row_factors[:, group_indices].mul_(column_factors)


def _decoder_inputs(shape, dtype, generator):
    shape = tuple(shape)
    if shape == (64,):
        exponent = -math.log(500000.0) / 64.0
        return torch.exp(torch.arange(64, dtype=torch.float32) * exponent).to(dtype)

    if shape == (4096,):
        return 0.75 + 0.5 * torch.rand(shape, dtype=dtype, generator=generator)

    if len(shape) == 2:
        return _structured_weight(shape, dtype, generator)

    return _uniform(shape, dtype, generator, 1.0)


def _bf16(value):
    return value.to(torch.bfloat16).float()


def _structured_linear(value, weight):
    value = value.reshape(-1, value.shape[-1]).float()
    weight = weight.float()
    output = torch.zeros(
        (value.shape[0], weight.shape[0]),
        dtype=torch.float32,
    )
    column_factors = weight[0]
    for group in range(_WEIGHT_GROUPS):
        begin = weight.shape[1] * group // _WEIGHT_GROUPS
        end = weight.shape[1] * (group + 1) // _WEIGHT_GROUPS
        row_factors = weight[:, begin] / column_factors[begin]
        reduction = torch.matmul(
            value[:, begin:end],
            column_factors[begin:end],
        )
        output.add_(reduction[:, None] * row_factors[None, :])
    return _bf16(output)


def _rms_norm(value, gamma):
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    normalized = _bf16(value * inverse_rms)
    return _bf16(normalized * gamma.reshape(1, 1, 4096))


def _rotate_half(value):
    return torch.cat((-value[..., 64:], value[..., :64]), dim=-1)


def _project(hidden, weight, heads):
    projected = _structured_linear(hidden, weight)
    return projected.reshape(hidden.shape[0], hidden.shape[1], heads, 128).permute(
        0, 2, 1, 3
    )


def _attention_residual(
    inv_freq,
    hidden,
    k_weight,
    gamma,
    key_cache,
    v_weight,
    value_cache,
    q_weight,
    o_weight,
):
    normalized = _rms_norm(hidden, gamma)

    sequence = hidden.shape[1]
    positions = torch.arange(sequence, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    angles = torch.cat((angles, angles), dim=-1)
    cosine = _bf16(torch.cos(angles)).reshape(1, 1, sequence, 128)
    sine = _bf16(torch.sin(angles)).reshape(1, 1, sequence, 128)

    key_update = _project(normalized, k_weight, 8)
    key_update = _bf16(
        _bf16(key_update * cosine) + _bf16(_rotate_half(key_update) * sine)
    )
    key_cache = key_cache.clone()
    key_cache[:, :, :sequence, :] = key_update

    value_update = _project(normalized, v_weight, 8)
    value_cache = value_cache.clone()
    value_cache[:, :, :sequence, :] = value_update

    query = _project(normalized, q_weight, 32)
    query = _bf16(_bf16(query * cosine) + _bf16(_rotate_half(query) * sine))

    key = key_cache.repeat_interleave(4, dim=1)
    value = value_cache.repeat_interleave(4, dim=1)
    scale = 0.297301769
    scores = torch.matmul(query * scale, (key * scale).transpose(-2, -1))
    causal = torch.arange(128).reshape(1, 1, 1, 128) <= positions.reshape(
        1, 1, sequence, 1
    )
    probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
    attention = _bf16(torch.matmul(probabilities, value))

    attention = attention.permute(0, 2, 1, 3).reshape(hidden.shape)
    attention_output = _structured_linear(attention, o_weight).reshape(hidden.shape)
    return _bf16(hidden + attention_output)


def _attention_residual_golden(
    inv_freq,
    hidden,
    k_weight,
    gamma,
    key_cache,
    v_weight,
    value_cache,
    q_weight,
    o_weight,
    post_gamma,
    gate_weight,
    up_weight,
    down_weight,
):
    del post_gamma, gate_weight, up_weight, down_weight
    return _attention_residual(
        inv_freq,
        hidden,
        k_weight,
        gamma,
        key_cache,
        v_weight,
        value_cache,
        q_weight,
        o_weight,
    )


def _decoder_intermediates(
    inv_freq,
    hidden,
    k_weight,
    gamma,
    key_cache,
    v_weight,
    value_cache,
    q_weight,
    o_weight,
    post_gamma,
    gate_weight,
    up_weight,
    down_weight,
):
    attention_residual = _attention_residual(
        inv_freq,
        hidden,
        k_weight,
        gamma,
        key_cache,
        v_weight,
        value_cache,
        q_weight,
        o_weight,
    )

    normalized = _rms_norm(attention_residual, post_gamma)
    gate_projection = _structured_linear(normalized, gate_weight)
    gate = _bf16(gate_projection * torch.sigmoid(gate_projection))
    up = _structured_linear(normalized, up_weight)
    gated = _bf16(gate * up)
    down = _structured_linear(gated, down_weight).reshape(hidden.shape)
    output = _bf16(attention_residual + down)
    return {
        "attention_residual": attention_residual,
        "normalized": normalized,
        "gate_projection": gate_projection.reshape(
            hidden.shape[0], hidden.shape[1], 14336
        ),
        "gate": gate.reshape(hidden.shape[0], hidden.shape[1], 14336),
        "up": up.reshape(hidden.shape[0], hidden.shape[1], 14336),
        "gated": gated.reshape(hidden.shape[0], hidden.shape[1], 14336),
        "down": down,
        "output": output,
    }


def _decoder_golden(*args):
    return _decoder_intermediates(*args)["output"]


PATTERN_TESTS = [
    PatternTest(
        name="llama_prefill_attention_residual_composed_e2e",
        ttir=_MODEL_TEXT.replace(
            "return %151, %221, %336 : tensor<32x8x128x128xbf16>, tensor<32x8x128x128xbf16>, tensor<32x18x4096xbf16>",
            "return %151, %221, %296 : tensor<32x8x128x128xbf16>, tensor<32x8x128x128xbf16>, tensor<32x18x4096xbf16>",
        ),
        golden=_attention_residual_golden,
        inputs=InputSpec(_decoder_inputs, seed=0),
        pcc=0.95,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        pattern_files=(
            "llama_projection_matmul_to_kernel.py",
            "llama_prefill_sdpa_to_kernel.py",
        ),
        output_index=2,
    ),
    PatternTest(
        name="llama_prefill_decoder_composed_e2e",
        ttir=_MODEL_TEXT,
        check="""
        CHECK-LABEL: func.func @main
        CHECK-COUNT-8: d2m.generic
        CHECK: d2m.tile_reduce_max
        CHECK: d2m.tile_exp
        CHECK: d2m.tile_reduce_sum
        CHECK: d2m.tile_div
        CHECK-COUNT-4: d2m.generic
        CHECK: return
        """,
        golden=_decoder_golden,
        inputs=InputSpec(_decoder_inputs, seed=0),
        pcc=0.95,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        pattern_files=(
            "llama_projection_matmul_to_kernel.py",
            "llama_prefill_sdpa_to_kernel.py",
        ),
        output_index=2,
    ),
]
