# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Composed full-shape Llama prefill attention validation."""

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
    / "llama_3_8b_prefill_attention.mlir"
)


def _attention_inputs(shape, dtype, generator):
    if shape == (64,):
        exponent = -math.log(500000.0) / 64.0
        return torch.exp(torch.arange(64, dtype=torch.float32) * exponent).to(dtype)

    if shape == (4096,):
        return 0.75 + 0.5 * torch.rand(shape, dtype=dtype, generator=generator)

    if shape in ((1024, 4096), (4096, 4096)):
        return 0.04 * torch.rand(shape, dtype=dtype, generator=generator) - 0.02

    return 0.25 * torch.rand(shape, dtype=dtype, generator=generator) - 0.125


def _bf16(value):
    return value.to(torch.bfloat16).float()


def _rotate_half(value):
    return torch.cat((-value[..., 64:], value[..., :64]), dim=-1)


def _project(hidden, weight, heads):
    projected = _bf16(torch.matmul(hidden.reshape(576, 4096), weight.T))
    return projected.reshape(32, 18, heads, 128).permute(0, 2, 1, 3)


def _attention_golden(
    inv_freq, hidden, k_weight, gamma, key_cache, v_weight, value_cache, q_weight
):
    inverse_rms = torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    normalized = _bf16(hidden * inverse_rms)
    normalized = _bf16(normalized * gamma.reshape(1, 1, 4096))

    positions = torch.arange(18, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    angles = torch.cat((angles, angles), dim=-1)
    cosine = _bf16(torch.cos(angles)).reshape(1, 1, 18, 128)
    sine = _bf16(torch.sin(angles)).reshape(1, 1, 18, 128)

    key_update = _project(normalized, k_weight, 8)
    key_update = _bf16(
        _bf16(key_update * cosine) + _bf16(_rotate_half(key_update) * sine)
    )
    key_cache = key_cache.clone()
    key_cache[:, :, :18, :] = key_update

    value_update = _project(normalized, v_weight, 8)
    value_cache = value_cache.clone()
    value_cache[:, :, :18, :] = value_update

    query = _project(normalized, q_weight, 32)
    query = _bf16(_bf16(query * cosine) + _bf16(_rotate_half(query) * sine))

    key = key_cache.repeat_interleave(4, dim=1)
    value = value_cache.repeat_interleave(4, dim=1)
    scale = 0.297301769
    scores = torch.matmul(query * scale, (key * scale).transpose(-2, -1))
    causal = torch.arange(128).reshape(1, 1, 1, 128) <= positions.reshape(1, 1, 18, 1)
    probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
    return _bf16(torch.matmul(probabilities, value))


PATTERN_TESTS = [
    PatternTest(
        name="llama_prefill_attention_composed_e2e",
        ttir=_MODEL.read_text(),
        check="""
        CHECK-LABEL: func.func @main
        CHECK: d2m.generic {{.*}}grid = #ttcore.grid<6x8>
        CHECK: d2m.generic {{.*}}grid = #ttcore.grid<8x8>
        CHECK: d2m.generic {{.*}}grid = #ttcore.grid<8x8>
        CHECK: d2m.generic {{.*}}grid = #ttcore.grid<6x8>
        CHECK: d2m.generic {{.*}}grid = #ttcore.grid<8x8>
        CHECK: d2m.generic {{.*}}grid = #ttcore.grid<8x8>
        CHECK: d2m.generic {{.*}}grid = #ttcore.grid<6x8>
        CHECK: d2m.generic {{.*}}grid = #ttcore.grid<8x4>
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_reduce_max
        CHECK: d2m.tile_exp
        CHECK: d2m.tile_reduce_sum
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_bcast
        CHECK: d2m.tile_div
        CHECK-NOT: d2m.generic
        CHECK: return
        """,
        golden=_attention_golden,
        inputs=InputSpec(_attention_inputs, seed=0),
        pcc=0.98,
        use_tile_matmul=False,
        e2e=True,
        pattern_files=(
            "llama_projection_matmul_to_kernel.py",
            "llama_prefill_sdpa_to_kernel.py",
        ),
        output_index=2,
    ),
]
