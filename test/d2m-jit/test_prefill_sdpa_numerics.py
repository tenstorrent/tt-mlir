# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host validation for the tile-streamed prefill SDPA recurrence."""

import math

import torch

_TILE = 32


def _identity(value):
    return value


def _round_bf16(value):
    return value.to(torch.bfloat16).float()


def _pcc(lhs, rhs):
    values = torch.stack((lhs.flatten().float(), rhs.flatten().float()))
    return torch.corrcoef(values)[0, 1].item()


def _golden_sdpa(query, key, value, mask, scale):
    heads_per_kv = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(heads_per_kv, dim=1)
    value = value.repeat_interleave(heads_per_kv, dim=1)
    scores = query @ key.transpose(-2, -1)
    probabilities = torch.softmax(scores * scale + mask, dim=-1)
    return probabilities @ value


def _streamed_sdpa(query, key, value, mask, scale, quantize):
    batch, query_heads, query_sequence, head_dim = query.shape
    _, kv_heads, key_sequence, _ = key.shape
    heads_per_kv = query_heads // kv_heads
    key = key.repeat_interleave(heads_per_kv, dim=1)
    value = value.repeat_interleave(heads_per_kv, dim=1)

    padded_query = math.ceil(query_sequence / _TILE) * _TILE
    padded_key = math.ceil(key_sequence / _TILE) * _TILE
    query_storage = torch.zeros(
        (batch, query_heads, padded_query, head_dim), dtype=torch.float32
    )
    key_storage = torch.zeros(
        (batch, query_heads, padded_key, head_dim), dtype=torch.float32
    )
    value_storage = torch.zeros_like(key_storage)
    query_storage[:, :, :query_sequence] = query
    key_storage[:, :, :key_sequence] = key
    value_storage[:, :, :key_sequence] = value

    mask = mask.expand(batch, query_heads, query_sequence, key_sequence)
    mask_storage = torch.zeros(
        (batch, query_heads, padded_query, padded_key), dtype=torch.float32
    )
    mask_storage[:, :, :query_sequence, :key_sequence] = mask
    key_padding = torch.zeros((_TILE, padded_key), dtype=torch.float32)
    key_padding[:, key_sequence:] = float("-inf")

    output = torch.zeros_like(query_storage)
    scale_value = quantize(torch.tensor(scale, dtype=torch.float32))
    negative_large = quantize(torch.full((_TILE, 1), -1.0e30, dtype=torch.float32))

    for batch_index in range(batch):
        for head_index in range(query_heads):
            for query_start in range(0, padded_query, _TILE):
                query_tile = query_storage[
                    batch_index,
                    head_index,
                    query_start : query_start + _TILE,
                ]
                row_max = negative_large.clone()
                row_sum = torch.zeros((_TILE, 1), dtype=torch.float32)
                output_accumulators = [
                    torch.zeros((_TILE, _TILE), dtype=torch.float32)
                    for _ in range(head_dim // _TILE)
                ]

                for key_start in range(0, padded_key, _TILE):
                    scores = torch.zeros((_TILE, _TILE), dtype=torch.float32)
                    for head_start in range(0, head_dim, _TILE):
                        query_head_tile = query_tile[:, head_start : head_start + _TILE]
                        key_head_tile = key_storage[
                            batch_index,
                            head_index,
                            key_start : key_start + _TILE,
                            head_start : head_start + _TILE,
                        ]
                        product = quantize(query_head_tile @ key_head_tile.T)
                        scores = quantize(scores + product)

                    scaled_scores = quantize(scores * scale_value)
                    mask_tile = mask_storage[
                        batch_index,
                        head_index,
                        query_start : query_start + _TILE,
                        key_start : key_start + _TILE,
                    ]
                    scaled_scores = quantize(scaled_scores + mask_tile)
                    padding_tile = key_padding[:, key_start : key_start + _TILE]
                    scaled_scores = quantize(scaled_scores + padding_tile)

                    block_max = quantize(scaled_scores.max(dim=1, keepdim=True).values)
                    next_row_max = quantize(torch.maximum(row_max, block_max))
                    previous_scale = quantize(
                        torch.exp(quantize(row_max - next_row_max))
                    )
                    probabilities = quantize(
                        torch.exp(quantize(scaled_scores - next_row_max))
                    )
                    block_sum = quantize(probabilities.sum(dim=1, keepdim=True))
                    row_sum = quantize(quantize(row_sum * previous_scale) + block_sum)

                    for output_tile, accumulator in enumerate(output_accumulators):
                        value_tile = value_storage[
                            batch_index,
                            head_index,
                            key_start : key_start + _TILE,
                            output_tile * _TILE : (output_tile + 1) * _TILE,
                        ]
                        block_output = quantize(probabilities @ value_tile)
                        output_accumulators[output_tile] = quantize(
                            quantize(accumulator * previous_scale) + block_output
                        )
                    row_max = next_row_max

                for output_tile, accumulator in enumerate(output_accumulators):
                    normalized = quantize(accumulator / row_sum)
                    normalized = torch.where(
                        row_sum > 0, normalized, torch.zeros_like(normalized)
                    )
                    output[
                        batch_index,
                        head_index,
                        query_start : query_start + _TILE,
                        output_tile * _TILE : (output_tile + 1) * _TILE,
                    ] = normalized

    return output[:, :, :query_sequence]


def _make_inputs(batch, query_heads, kv_heads, query_sequence, key_sequence, head_dim):
    generator = torch.Generator().manual_seed(11)
    shapes = (
        (batch, query_heads, query_sequence, head_dim),
        (batch, kv_heads, key_sequence, head_dim),
        (batch, kv_heads, key_sequence, head_dim),
    )
    return tuple(
        _round_bf16(
            torch.rand(shape, generator=generator, dtype=torch.float32) * 0.5 - 0.25
        )
        for shape in shapes
    )


def _check_recurrence(query, key, value, mask, scale):
    golden = _golden_sdpa(query, key, value, mask, scale)
    fp32_streamed = _streamed_sdpa(query, key, value, mask, scale, _identity)
    torch.testing.assert_close(fp32_streamed, golden, atol=2.0e-6, rtol=2.0e-5)

    bf16_streamed = _streamed_sdpa(query, key, value, mask, scale, _round_bf16)
    correlation = _pcc(bf16_streamed, golden)
    assert correlation >= 0.999, f"BF16 streamed SDPA PCC {correlation}"


def test_streamed_sdpa_rectangular_gqa_additive_mask():
    query, key, value = _make_inputs(2, 4, 2, 45, 80, 64)
    generator = torch.Generator().manual_seed(29)
    mask = _round_bf16(
        torch.rand((1, 4, 45, 80), generator=generator, dtype=torch.float32) * 0.1
        - 0.05
    )
    mask[:, :, :, -5:] = float("-inf")
    _check_recurrence(query, key, value, mask, scale=0.125)


def test_streamed_sdpa_causal_mqa():
    query, key, value = _make_inputs(1, 4, 1, 64, 64, 32)
    mask = torch.triu(
        torch.full((1, 1, 64, 64), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )
    _check_recurrence(query, key, value, mask, scale=1.0 / math.sqrt(32))
