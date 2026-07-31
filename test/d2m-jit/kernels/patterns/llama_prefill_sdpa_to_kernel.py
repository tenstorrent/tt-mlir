# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused D2M-JIT prefill attention lowerings."""

import math
from dataclasses import dataclass

import torch

import d2m_jit as d2m
from ttmlir import ir
from ttmlir.dialects import ttir

try:
    from runner import InputSpec, PatternTest
except ModuleNotFoundError as exc:
    if exc.name != "runner":
        raise

    class PatternTest:
        def __init__(self, **kwargs):
            pass

    class InputSpec:
        def __init__(self, *args, **kwargs):
            pass


_BATCH = 32
_QUERY_HEADS = 32
_KV_HEADS = 8
_SEQUENCE = 18
_HEAD_DIM = 128
_CACHE_SEQUENCE = 128
_PHYSICAL_GRID = 8
_MAX_SINGLE_TILE_SEQUENCE = 32


def _largest_grid_divisor(tile_count):
    for candidate in range(min(_PHYSICAL_GRID, tile_count), 0, -1):
        if tile_count % candidate == 0:
            return candidate
    raise AssertionError("positive tile count must have a grid divisor")


def _largest_common_grid_divisor(*tile_counts):
    for candidate in range(min(_PHYSICAL_GRID, *tile_counts), 0, -1):
        if all(tile_count % candidate == 0 for tile_count in tile_counts):
            return candidate
    raise AssertionError("positive tile counts must have a common grid divisor")


@d2m.kernel
def llama_prefill_rope_fused(
    input, cos, sin, out, batch_blocks, head_blocks, physical_grid, half_tiles
):
    batch_start = core_index(0)
    head_start = core_index(1)

    for batch_block in range(batch_blocks):
        batch = batch_start + batch_block * physical_grid
        for head_block in range(head_blocks):
            head = head_start + head_block * physical_grid
            for tile in range(half_tiles):
                x = typecast(
                    remote_load(input, [batch, head, 0, tile]),
                    "fp32",
                )
                rotated = typecast(
                    remote_load(input, [batch, head, 0, tile + half_tiles]),
                    "fp32",
                )
                cos_tile = typecast(
                    remote_load(cos, [0, 0, 0, tile]),
                    "fp32",
                )
                sin_tile = typecast(
                    remote_load(sin, [0, 0, 0, tile]),
                    "fp32",
                )
                result = typecast((x * cos_tile) + ((-rotated) * sin_tile), "bf16")
                remote_store(out, [batch, head, 0, tile], result)

            for tile in range(half_tiles):
                output_tile = tile + half_tiles
                x = typecast(
                    remote_load(input, [batch, head, 0, output_tile]),
                    "fp32",
                )
                rotated = typecast(
                    remote_load(input, [batch, head, 0, tile]),
                    "fp32",
                )
                cos_tile = typecast(
                    remote_load(cos, [0, 0, 0, output_tile]),
                    "fp32",
                )
                sin_tile = typecast(
                    remote_load(sin, [0, 0, 0, output_tile]),
                    "fp32",
                )
                result = typecast((x * cos_tile) + (rotated * sin_tile), "bf16")
                remote_store(out, [batch, head, 0, output_tile], result)


def _matches_llama_prefill_rope(op):
    if len(op.operands) != 3:
        return False
    try:
        composite_name = ir.StringAttr(op.attributes["composite_name"]).value
        input_type = ir.RankedTensorType(op.operands[0].type)
        cos_type = ir.RankedTensorType(op.operands[1].type)
        sin_type = ir.RankedTensorType(op.operands[2].type)
        result_type = ir.RankedTensorType(op.results[0].type)
    except (KeyError, TypeError, ValueError):
        return False

    input_shape = tuple(input_type.shape)
    return (
        composite_name == "rotary_embedding"
        and input_shape
        in {
            (_BATCH, _KV_HEADS, _SEQUENCE, _HEAD_DIM),
            (_BATCH, _QUERY_HEADS, _SEQUENCE, _HEAD_DIM),
        }
        and tuple(cos_type.shape) == (1, 1, _SEQUENCE, _HEAD_DIM)
        and tuple(sin_type.shape) == (1, 1, _SEQUENCE, _HEAD_DIM)
        and tuple(result_type.shape) == input_shape
        and all(
            ir.BF16Type.isinstance(t.element_type)
            for t in (input_type, cos_type, sin_type, result_type)
        )
    )


@d2m.pattern(
    root="ttcore.composite",
    benefit=200,
    match=_matches_llama_prefill_rope,
)
def lower_llama_prefill_rope(op, rewriter):
    input_shape = tuple(ir.RankedTensorType(op.operands[0].type).shape)
    heads = input_shape[1]
    head_tiles = _HEAD_DIM // 32
    input_layout = d2m.Layout(
        shape=input_shape,
        dtype=d2m.bfloat16,
        block_shape=[1, 1, 1, 1],
        grid_shape=[_PHYSICAL_GRID, _PHYSICAL_GRID, 1, 1],
        collapse=False,
        mem_space="dram",
    )
    embedding_layout = d2m.Layout(
        shape=(1, 1, _SEQUENCE, _HEAD_DIM),
        dtype=d2m.bfloat16,
        block_shape=[1, 1, 1, 1],
        grid_shape=[1, 1, 1, head_tiles],
        collapse=False,
        mem_space="dram",
    )

    input = d2m.to_layout(d2m.from_value(op.operands[0]), input_layout)
    cos = d2m.to_layout(d2m.from_value(op.operands[1]), embedding_layout)
    sin = d2m.to_layout(d2m.from_value(op.operands[2]), embedding_layout)
    compute_out = d2m.empty(input_layout)
    llama_prefill_rope_fused(
        input,
        cos,
        sin,
        compute_out,
        _BATCH // _PHYSICAL_GRID,
        heads // _PHYSICAL_GRID,
        _PHYSICAL_GRID,
        head_tiles // 2,
        grid=(_PHYSICAL_GRID, _PHYSICAL_GRID),
    )
    return d2m.from_device(compute_out)


@d2m.kernel
def llama_prefill_fill_cache_update(
    cache,
    input,
    mask,
    out,
    out_physical,
    batch_blocks,
    physical_grid,
    head_tiles,
):
    batch_start = core_index(0)
    head = core_index(1)

    for batch_block in range(batch_blocks):
        batch = batch_start + batch_block * physical_grid
        for head_tile in range(head_tiles):
            mask_tile = remote_load(mask, [0, 0, 0, 0])
            cache_tile = remote_load(cache, [batch, head, 0, head_tile])
            input_tile = remote_load(input, [batch, head, 0, head_tile])
            result = where(gtz(mask_tile), input_tile, cache_tile)
            remote_store(out, [batch, head, 0, head_tile], result)


@d2m.kernel
def llama_prefill_fill_cache_copy(
    cache,
    out,
    out_physical,
    batch_blocks,
    physical_grid,
    head_tiles,
    cache_sequence_tiles,
):
    batch_start = core_index(0)
    head = core_index(1)

    for batch_block in range(batch_blocks):
        batch = batch_start + batch_block * physical_grid
        for sequence_tile in range(1, cache_sequence_tiles):
            for head_tile in range(head_tiles):
                cache_tile = remote_load(cache, [batch, head, sequence_tile, head_tile])
                remote_store(
                    out,
                    [batch, head, sequence_tile, head_tile],
                    cache_tile,
                )


def _matches_llama_prefill_fill_cache(op):
    if len(op.operands) != 2:
        return False
    try:
        batch_offset = int(ir.IntegerAttr(op.attributes["batch_offset"]).value)
        cache_type = ir.RankedTensorType(op.operands[0].type)
        input_type = ir.RankedTensorType(op.operands[1].type)
        result_type = ir.RankedTensorType(op.results[0].type)
    except (KeyError, TypeError, ValueError):
        return False

    cache_shape = (_BATCH, _KV_HEADS, _CACHE_SEQUENCE, _HEAD_DIM)
    input_shape = tuple(input_type.shape)
    return (
        batch_offset == 0
        and tuple(cache_type.shape) == cache_shape
        and len(input_shape) == 4
        and input_shape[0] == _BATCH
        and input_shape[1] == _KV_HEADS
        and 0 < input_shape[2] <= _MAX_SINGLE_TILE_SEQUENCE
        and input_shape[3] == _HEAD_DIM
        and tuple(result_type.shape) == cache_shape
        and all(
            ir.BF16Type.isinstance(t.element_type)
            for t in (cache_type, input_type, result_type)
        )
    )


def _integer_array_attr(op, name):
    return tuple(int(element.value) for element in op.attributes[name])


def _match_terminal_llama_prefill_fill_cache_chain(op):
    cache_shape = (_BATCH, _KV_HEADS, _CACHE_SEQUENCE, _HEAD_DIM)
    terminal_slice_shape = _shape(op.operands[1])
    if (
        terminal_slice_shape is None
        or len(terminal_slice_shape) != 4
        or terminal_slice_shape[0] != 1
        or terminal_slice_shape[1] != _KV_HEADS
        or not 0 < terminal_slice_shape[2] <= _MAX_SINGLE_TILE_SEQUENCE
        or terminal_slice_shape[3] != _HEAD_DIM
    ):
        return None

    sequence = terminal_slice_shape[2]
    input_shape = (_BATCH, _KV_HEADS, sequence, _HEAD_DIM)
    slice_shape = (1, _KV_HEADS, sequence, _HEAD_DIM)

    current = op
    input_value = None
    for batch_offset in range(_BATCH - 1, -1, -1):
        if (
            current is None
            or current.name != "ttir.fill_cache"
            or len(current.operands) != 2
            or len(current.results) != 1
            or int(ir.IntegerAttr(current.attributes["batch_offset"]).value)
            != batch_offset
            or _shape(current.operands[0]) != cache_shape
            or _shape(current.operands[1]) != slice_shape
            or _shape(current.results[0]) != cache_shape
        ):
            return None

        slice_op = current.operands[1].owner
        if (
            slice_op is None
            or slice_op.name != "ttir.slice_static"
            or len(slice_op.operands) != 1
            or _integer_array_attr(slice_op, "begins") != (batch_offset, 0, 0, 0)
            or _integer_array_attr(slice_op, "ends")
            != (batch_offset + 1, _KV_HEADS, sequence, _HEAD_DIM)
            or _integer_array_attr(slice_op, "step") != (1, 1, 1, 1)
        ):
            return None

        slice_input = slice_op.operands[0]
        if _shape(slice_input) != input_shape:
            return None
        if input_value is None:
            input_value = slice_input
        elif slice_input != input_value:
            return None

        if batch_offset == 0:
            cache_value = current.operands[0]
        else:
            current = current.operands[0].owner

    if not all(
        ir.BF16Type.isinstance(ir.RankedTensorType(value.type).element_type)
        for value in (cache_value, input_value, op.results[0])
    ):
        return None
    return cache_value, input_value


def _matches_terminal_llama_prefill_fill_cache_chain(op):
    try:
        return _match_terminal_llama_prefill_fill_cache_chain(op) is not None
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def _lower_llama_prefill_fill_cache(cache_value, input_value):
    sequence = _shape(input_value)[2]
    head_tiles = _HEAD_DIM // 32
    cache_sequence_tiles = _CACHE_SEQUENCE // 32
    cache_layout = d2m.Layout(
        shape=(_BATCH, _KV_HEADS, _CACHE_SEQUENCE, _HEAD_DIM),
        dtype=d2m.bfloat16,
        block_shape=[1, 1, 1, 1],
        grid_shape=[_PHYSICAL_GRID, _PHYSICAL_GRID, 1, 1],
        collapse=False,
        mem_space="dram",
    )
    input_layout = d2m.Layout(
        shape=(_BATCH, _KV_HEADS, sequence, _HEAD_DIM),
        dtype=d2m.bfloat16,
        block_shape=[1, 1, 1, 1],
        grid_shape=[_PHYSICAL_GRID, _PHYSICAL_GRID, 1, 1],
        collapse=False,
        mem_space="dram",
    )
    mask_layout = d2m.Layout(
        shape=(1, 1, 32, 32),
        dtype=d2m.bfloat16,
        block_shape=[1, 1, 1, 1],
        grid_shape=[1, 1, 1, 1],
        collapse=False,
    )

    fill_mask = torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16)
    fill_mask[:, :, :sequence, :] = 1

    cache = d2m.to_layout(d2m.from_value(cache_value), cache_layout)
    input = d2m.to_layout(d2m.from_value(input_value), input_layout)
    mask = d2m.to_layout(fill_mask, mask_layout)
    compute_out = d2m.empty(cache_layout)
    out_physical = d2m.from_value(compute_out.unblocked_value, cache_layout)
    llama_prefill_fill_cache_update(
        cache,
        input,
        mask,
        compute_out,
        out_physical,
        _BATCH // _PHYSICAL_GRID,
        _PHYSICAL_GRID,
        head_tiles,
        grid=(_PHYSICAL_GRID, _PHYSICAL_GRID),
        num_outs=2,
    )
    llama_prefill_fill_cache_copy(
        cache,
        compute_out,
        out_physical,
        _BATCH // _PHYSICAL_GRID,
        _PHYSICAL_GRID,
        head_tiles,
        cache_sequence_tiles,
        grid=(_PHYSICAL_GRID, _PHYSICAL_GRID),
        num_outs=2,
    )
    return d2m.from_device(out_physical, physical_storage=True)


@d2m.pattern(
    root=ttir.FillCacheOp,
    benefit=250,
    match=_matches_terminal_llama_prefill_fill_cache_chain,
)
def lower_terminal_llama_prefill_fill_cache_chain(op, rewriter):
    cache_value, input_value = _match_terminal_llama_prefill_fill_cache_chain(op)
    return _lower_llama_prefill_fill_cache(cache_value, input_value)


@d2m.pattern(
    root=ttir.FillCacheOp,
    benefit=150,
    match=_matches_llama_prefill_fill_cache,
)
def lower_llama_prefill_fill_cache(op, rewriter):
    return _lower_llama_prefill_fill_cache(op.operands[0], op.operands[1])


@d2m.kernel
def prefill_sdpa_fused(
    q,
    k,
    v,
    mask,
    key_padding_mask,
    scale,
    negative_large,
    out,
    query_row_blocks,
    physical_grid_rows,
    physical_grid_cols,
    query_heads,
    kv_heads,
    query_heads_per_kv,
    query_sequence_tiles,
    key_sequence_tiles,
    head_tiles,
    output_head_blocks,
    mask_batch_stride,
    mask_head_stride,
):
    query_row_start = core_index(0)
    output_head_start = core_index(1)

    for query_row_block in range(query_row_blocks):
        query_row = query_row_start + query_row_block * physical_grid_rows
        query_rows_per_batch = query_heads * query_sequence_tiles
        batch = query_row // query_rows_per_batch
        query_head_and_tile = query_row % query_rows_per_batch
        query_head = query_head_and_tile // query_sequence_tiles
        query_sequence_tile = query_head_and_tile % query_sequence_tiles
        kv_head = query_head // query_heads_per_kv
        kv_row_start = (batch * kv_heads + kv_head) * key_sequence_tiles

        for output_head_block in range(output_head_blocks):
            output_head_tile = (
                output_head_start + output_head_block * physical_grid_cols
            )
            row_max_state = remote_load(negative_large, [0, 0])
            row_sum_state = zeros([1, 1], dtype="bf16")
            output_acc_state = zeros([1, 1], dtype="bf16")

            for key_sequence_tile in range(key_sequence_tiles):
                disable_l1_accumulation()
                scores = zeros([1, 1], dtype="bf16")
                for reduction_head_tile in range(head_tiles):
                    query_tile = remote_load(q, [query_row, reduction_head_tile])
                    key_tile = remote_load(
                        k,
                        [kv_row_start + key_sequence_tile, reduction_head_tile],
                    )
                    scores = matmul(
                        query_tile,
                        key_tile,
                        transpose_b=True,
                        acc=scores,
                    )

                scale_tile = remote_load(scale, [0, 0])
                mask_row = (
                    batch * mask_batch_stride
                    + query_head * mask_head_stride
                    + query_sequence_tile
                )
                mask_tile = remote_load(mask, [mask_row, key_sequence_tile])
                key_padding_tile = remote_load(key_padding_mask, [0, key_sequence_tile])
                scaled_scores = scores * scale_tile + mask_tile + key_padding_tile
                block_max = reduce_max(scaled_scores, 1)
                row_max = maximum(row_max_state, block_max)
                previous_scale = exp(row_max_state - row_max)
                probabilities = exp(scaled_scores - row_max)
                block_sum = reduce_sum(probabilities, 1)
                row_sum_state = row_sum_state * previous_scale + block_sum

                value_tile = remote_load(
                    v,
                    [kv_row_start + key_sequence_tile, output_head_tile],
                )
                block_output = probabilities @ value_tile
                output_acc_state = (
                    output_acc_state * tile_bcast_col(previous_scale) + block_output
                )
                row_max_state = row_max

            denominator = tile_bcast_col(row_sum_state)
            normalized = output_acc_state / denominator
            result = where(
                gtz(denominator),
                normalized,
                zeros([1, 1], dtype="bf16"),
            )
            remote_store(out, [query_row, output_head_tile], result)


@dataclass(frozen=True)
class PrefillSDPA:
    """Canonical inputs and semantics for the streamed prefill kernel."""

    query: object
    key: object
    value: object
    mask: object
    scale: float
    is_causal: bool
    query_shape: tuple
    key_shape: tuple
    mask_shape: tuple | None


def _is_bf16(value):
    return ir.BF16Type.isinstance(ir.RankedTensorType(value.type).element_type)


def _make_prefill_sdpa(query, key, value, mask, scale, is_causal):
    try:
        query_shape = tuple(ir.RankedTensorType(query.type).shape)
        key_shape = tuple(ir.RankedTensorType(key.type).shape)
        value_shape = tuple(ir.RankedTensorType(value.type).shape)
        mask_shape = None
        if mask is not None:
            mask_shape = tuple(ir.RankedTensorType(mask.type).shape)
    except (TypeError, ValueError):
        return None

    if not all(len(shape) == 4 for shape in (query_shape, key_shape, value_shape)):
        return None
    batch, query_heads, query_sequence, head_dim = query_shape
    key_batch, kv_heads, key_sequence, key_head_dim = key_shape
    if (
        value_shape != key_shape
        or key_batch != batch
        or key_head_dim != head_dim
        or query_sequence <= 1
        or min(batch, query_heads, kv_heads, key_sequence, head_dim) <= 0
        or query_heads % kv_heads != 0
        or head_dim % 32 != 0
        or not all(_is_bf16(v) for v in (query, key, value))
    ):
        return None

    if is_causal:
        if mask is not None or query_sequence != key_sequence:
            return None
    elif mask is not None:
        if not _is_bf16(mask) or mask_shape is None or len(mask_shape) != 4:
            return None
        mask_batch, mask_heads, mask_query, mask_key = mask_shape
        if (
            mask_batch not in (1, batch)
            or mask_heads not in (1, query_heads)
            or mask_query != query_sequence
            or mask_key != key_sequence
        ):
            return None

    return PrefillSDPA(
        query=query,
        key=key,
        value=value,
        mask=mask,
        scale=float(scale),
        is_causal=is_causal,
        query_shape=query_shape,
        key_shape=key_shape,
        mask_shape=mask_shape,
    )


def _lower_prefill_sdpa(match):
    batch, query_heads, query_sequence, head_dim = match.query_shape
    _, kv_heads, key_sequence, _ = match.key_shape
    query_sequence_tiles = (query_sequence + 31) // 32
    key_sequence_tiles = (key_sequence + 31) // 32
    padded_key = key_sequence_tiles * 32
    head_tiles = head_dim // 32
    query_rows = batch * query_heads * query_sequence_tiles
    key_rows = batch * kv_heads * key_sequence_tiles

    if match.mask is None:
        grid_rows = _largest_common_grid_divisor(query_rows, key_rows)
    else:
        mask_rows = match.mask_shape[0] * match.mask_shape[1] * query_sequence_tiles
        grid_rows = _largest_common_grid_divisor(query_rows, key_rows, mask_rows)
    grid_cols = _largest_grid_divisor(head_tiles)

    query_layout = d2m.Layout(
        shape=match.query_shape,
        dtype=d2m.bfloat16,
        block_shape=[1, 1, 1, 1],
        grid_shape=[grid_rows, grid_cols],
        mem_space="dram",
    )
    kv_layout = d2m.Layout(
        shape=match.key_shape,
        dtype=d2m.bfloat16,
        block_shape=[1, 1, 1, 1],
        grid_shape=[grid_rows, grid_cols],
        mem_space="dram",
    )
    tile_layout = d2m.Layout(
        shape=(32, 32),
        dtype=d2m.bfloat16,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )

    query = d2m.to_layout(d2m.from_value(match.query), query_layout)
    key = d2m.to_layout(d2m.from_value(match.key), kv_layout)
    value = d2m.to_layout(d2m.from_value(match.value), kv_layout)

    if match.mask is not None:
        mask_layout = d2m.Layout(
            shape=match.mask_shape,
            dtype=d2m.bfloat16,
            block_shape=[1, 1, 1, 1],
            grid_shape=[grid_rows, _largest_grid_divisor(key_sequence_tiles)],
            mem_space="dram",
        )
        mask = d2m.to_layout(d2m.from_value(match.mask), mask_layout)
        mask_batch, mask_heads, _, _ = match.mask_shape
        mask_batch_stride = 0 if mask_batch == 1 else mask_heads * query_sequence_tiles
        mask_head_stride = 0 if mask_heads == 1 else query_sequence_tiles
    else:
        padded_query = query_sequence_tiles * 32
        mask_tensor = torch.zeros((padded_query, padded_key), dtype=torch.float32)
        if match.is_causal:
            mask_tensor.fill_(float("-inf"))
            mask_tensor[:query_sequence, :key_sequence] = torch.triu(
                torch.full(
                    (query_sequence, key_sequence),
                    float("-inf"),
                    dtype=torch.float32,
                ),
                diagonal=1,
            )
        elif key_sequence < padded_key:
            mask_tensor[:, key_sequence:] = float("-inf")
        mask_layout = d2m.Layout(
            shape=mask_tensor.shape,
            dtype=d2m.bfloat16,
            block_shape=[1, 1],
            grid_shape=[
                _largest_grid_divisor(query_sequence_tiles),
                _largest_grid_divisor(key_sequence_tiles),
            ],
        )
        mask = d2m.to_layout(mask_tensor, mask_layout)
        mask_batch_stride = 0
        mask_head_stride = 0

    key_padding_tensor = torch.zeros((32, padded_key), dtype=torch.float32)
    if key_sequence < padded_key:
        key_padding_tensor[:, key_sequence:] = float("-inf")
    key_padding_layout = d2m.Layout(
        shape=key_padding_tensor.shape,
        dtype=d2m.bfloat16,
        block_shape=[1, 1],
        grid_shape=[1, _largest_grid_divisor(key_sequence_tiles)],
    )
    key_padding_mask = d2m.to_layout(key_padding_tensor, key_padding_layout)

    scale = d2m.to_layout(
        torch.full((32, 32), match.scale, dtype=torch.float32), tile_layout
    )
    negative_large = d2m.to_layout(
        torch.full((32, 32), -1.0e30, dtype=torch.float32), tile_layout
    )
    compute_out = d2m.empty(query_layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = False
    try:
        prefill_sdpa_fused(
            query,
            key,
            value,
            mask,
            key_padding_mask,
            scale,
            negative_large,
            compute_out,
            query_rows // grid_rows,
            grid_rows,
            grid_cols,
            query_heads,
            kv_heads,
            query_heads // kv_heads,
            query_sequence_tiles,
            key_sequence_tiles,
            head_tiles,
            head_tiles // grid_cols,
            mask_batch_stride,
            mask_head_stride,
            grid=(grid_rows, grid_cols),
        )
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul

    return d2m.from_device(compute_out)


def _match_scaled_dot_product_attention(op):
    try:
        segments = tuple(int(value) for value in op.attributes["operandSegmentSizes"])
        if len(segments) != 5 or segments[:3] != (1, 1, 1):
            return None
        if segments[4] != 0 or "sliding_window_size" in op.attributes:
            return None
        expected_operands = 3 + segments[3] + segments[4]
        if len(op.operands) != expected_operands or len(op.results) != 1:
            return None

        query, key, value = op.operands[:3]
        mask = op.operands[3] if segments[3] else None
        is_causal = True
        if "is_causal" in op.attributes:
            is_causal = bool(ir.BoolAttr(op.attributes["is_causal"]).value)
        head_dim = tuple(ir.RankedTensorType(query.type).shape)[-1]
        scale = 1.0 / math.sqrt(head_dim)
        if "scale" in op.attributes:
            scale = float(ir.FloatAttr(op.attributes["scale"]).value)
        match = _make_prefill_sdpa(query, key, value, mask, scale, is_causal)
        if match is None:
            return None
        result_shape = tuple(ir.RankedTensorType(op.results[0].type).shape)
        if result_shape != match.query_shape or not _is_bf16(op.results[0]):
            return None
        return match
    except (IndexError, KeyError, TypeError, ValueError):
        return None


@d2m.pattern(
    root=ttir.ScaledDotProductAttentionOp,
    benefit=100,
    match=lambda op: _match_scaled_dot_product_attention(op) is not None,
)
def lower_scaled_dot_product_attention(op, rewriter):
    return _lower_prefill_sdpa(_match_scaled_dot_product_attention(op))


def _producer(value, name):
    owner = value.owner
    names = (name,) if isinstance(name, str) else name
    if getattr(owner, "name", None) not in names:
        return None
    return owner


def _shape(value):
    try:
        return tuple(ir.RankedTensorType(value.type).shape)
    except (TypeError, ValueError):
        return None


_ATTENTION_VIEW_OPS = (
    "ttir.broadcast",
    "ttir.reshape",
    "ttir.typecast",
)


def _only_unit_dims_changed(source_shape, result_shape):
    if math.prod(source_shape) != math.prod(result_shape):
        return False
    return tuple(dim for dim in source_shape if dim != 1) == tuple(
        dim for dim in result_shape if dim != 1
    )


def _is_broadcast_shape(source_shape, result_shape):
    if len(source_shape) > len(result_shape):
        return False
    padded_source = (1,) * (len(result_shape) - len(source_shape)) + source_shape
    return all(
        source_dim in (1, result_dim)
        for source_dim, result_dim in zip(padded_source, result_shape)
    )


def _strip_attention_views(value):
    """Strip only views that preserve element order or broadcast singleton axes."""

    while True:
        owner = value.owner
        owner_name = getattr(owner, "name", None)
        if owner_name not in _ATTENTION_VIEW_OPS or len(owner.operands) != 1:
            return value

        source = owner.operands[0]
        source_shape = _shape(source)
        result_shape = _shape(value)
        if source_shape is None or result_shape is None:
            return None
        if owner_name == "ttir.typecast" and source_shape != result_shape:
            return None
        if owner_name == "ttir.reshape" and not _only_unit_dims_changed(
            source_shape, result_shape
        ):
            return None
        if owner_name == "ttir.broadcast" and not _is_broadcast_shape(
            source_shape, result_shape
        ):
            return None
        value = source


def _strip_uniform_views(value):
    while getattr(value.owner, "name", None) in _ATTENTION_VIEW_OPS:
        value = value.owner.operands[0]
    return value


def _uniform_constant(value):
    value = _strip_uniform_views(value)
    owner = value.owner
    owner_name = getattr(owner, "name", None)
    if owner_name is None:
        return None
    if owner_name == "ttir.zeros":
        return 0.0
    if owner_name != "ttir.full":
        return None
    try:
        return float(ir.FloatAttr(owner.attributes["fill_value"]).value)
    except (KeyError, TypeError, ValueError):
        return None


def _transpose_on_last_two_dims(op):
    if op.name != "ttir.permute":
        return False
    try:
        permutation = tuple(int(value) for value in op.attributes["permutation"])
    except (KeyError, TypeError, ValueError):
        return False
    rank = len(permutation)
    return permutation == tuple(range(rank - 2)) + (rank - 1, rank - 2)


@dataclass(frozen=True)
class _AttentionOperand:
    source: object
    scale: float
    transposed: bool
    presented_shape: tuple
    steps: tuple


def _analyze_attention_operand(value, allow_transpose=False):
    """Record an operand's path back to storage without assuming its meaning."""

    scale = 1.0
    transposed = False
    presented_shape = _shape(value)
    if presented_shape is None:
        return None
    reverse_steps = []
    while True:
        owner = value.owner
        owner_name = getattr(owner, "name", None)
        if owner_name in _ATTENTION_VIEW_OPS:
            source = owner.operands[0]
            reverse_steps.append((owner_name, _shape(source), _shape(value)))
            value = source
            continue
        if owner_name == "ttir.permute" and allow_transpose:
            if transposed or not _transpose_on_last_two_dims(owner):
                return None
            transposed = True
            source = owner.operands[0]
            reverse_steps.append((owner_name, _shape(source), _shape(value)))
            value = source
            continue
        if owner_name == "ttir.multiply":
            lhs_scale = _uniform_constant(owner.operands[0])
            rhs_scale = _uniform_constant(owner.operands[1])
            if (lhs_scale is None) == (rhs_scale is None):
                return None
            if lhs_scale is not None:
                scale *= lhs_scale
                source = owner.operands[1]
            else:
                scale *= rhs_scale
                source = owner.operands[0]
            reverse_steps.append(("scale", _shape(source), _shape(value)))
            value = source
            continue
        if owner_name == "ttir.div":
            divisor = _uniform_constant(owner.operands[1])
            if divisor is None or divisor == 0.0:
                return None
            scale /= divisor
            source = owner.operands[0]
            reverse_steps.append(("scale", _shape(source), _shape(value)))
            value = source
            continue
        if any(
            source_shape is None or result_shape is None
            for _, source_shape, result_shape in reverse_steps
        ):
            return None
        return _AttentionOperand(
            source=value,
            scale=scale,
            transposed=transposed,
            presented_shape=presented_shape,
            steps=tuple(reversed(reverse_steps)),
        )


def _valid_attention_path(operand, source_shape, expanded_shape, transpose=False):
    """Validate the canonical contiguous GQA head-expansion view."""

    if _shape(operand.source) != source_shape:
        return False
    if len(source_shape) != 4 or len(expanded_shape) != 4:
        return False

    batch, source_heads, sequence, head_dim = source_shape
    (
        expanded_batch,
        expanded_heads,
        expanded_sequence,
        expanded_head_dim,
    ) = expanded_shape
    if (
        expanded_batch != batch
        or expanded_sequence != sequence
        or expanded_head_dim != head_dim
        or expanded_heads % source_heads != 0
    ):
        return False

    head_group = expanded_heads // source_heads
    singleton_head_shape = (batch, source_heads, 1, sequence, head_dim)
    grouped_head_shape = (
        batch,
        source_heads,
        head_group,
        sequence,
        head_dim,
    )
    transposed_shape = expanded_shape[:-2] + (
        expanded_shape[-1],
        expanded_shape[-2],
    )
    expected_shape = transposed_shape if transpose else expanded_shape
    current_shape = source_shape
    saw_transpose = False

    for kind, step_source, step_result in operand.steps:
        if step_source != current_shape:
            return False
        if kind == "ttir.typecast":
            valid = step_result == current_shape
        elif kind == "ttir.reshape":
            valid = (current_shape, step_result) in {
                (source_shape, singleton_head_shape),
                (grouped_head_shape, expanded_shape),
            } or step_result == current_shape
        elif kind in ("ttir.broadcast", "scale"):
            valid = step_result == current_shape or (
                current_shape == singleton_head_shape
                and step_result == grouped_head_shape
            )
            if source_heads == 1:
                valid = valid or (
                    current_shape == source_shape and step_result == expanded_shape
                )
        elif kind == "ttir.permute":
            valid = (
                transpose
                and not saw_transpose
                and current_shape == expanded_shape
                and step_result == transposed_shape
            )
            saw_transpose = True
        else:
            return False
        if not valid:
            return False
        current_shape = step_result

    return (
        current_shape == expected_shape
        and operand.presented_shape == expected_shape
        and operand.transposed == saw_transpose == transpose
    )


def _valid_score_path(operand, score_shape):
    if operand.transposed or _shape(operand.source) != score_shape:
        return False
    return operand.presented_shape == score_shape and all(
        kind in _ATTENTION_VIEW_OPS + ("scale",)
        and source_shape == result_shape == score_shape
        for kind, source_shape, result_shape in operand.steps
    )


def _valid_mask_path(operand, score_shape):
    if operand.transposed or operand.scale != 1.0:
        return False
    current_shape = _shape(operand.source)
    if current_shape is None:
        return False
    for kind, source_shape, result_shape in operand.steps:
        if source_shape != current_shape:
            return False
        if kind == "ttir.typecast":
            valid = result_shape == source_shape
        elif kind == "ttir.reshape":
            valid = _only_unit_dims_changed(source_shape, result_shape)
        elif kind == "ttir.broadcast":
            valid = _is_broadcast_shape(source_shape, result_shape)
        else:
            valid = False
        if not valid:
            return False
        current_shape = result_shape
    return current_shape == operand.presented_shape and _is_broadcast_shape(
        current_shape, score_shape
    )


def _reduction_is_last_dim(op, input_value):
    try:
        dims = tuple(
            int(getattr(value, "value", value)) for value in op.attributes["dim_arg"]
        )
        rank = len(_shape(input_value))
    except (KeyError, TypeError, ValueError):
        return False
    return dims == (rank - 1,)


def _match_softmax_input(value):
    """Return the exact score tensor consumed by a stable last-dim softmax."""

    value = _strip_attention_views(value)
    if value is None:
        return None
    where = _producer(value, "ttir.where")
    if where is not None:
        if len(where.operands) != 3 or _uniform_constant(where.operands[1]) != 0.0:
            return None
        value = _strip_attention_views(where.operands[2])
        if value is None:
            return None

    softmax = _producer(value, "ttir.softmax")
    if softmax is not None:
        try:
            dimension = int(ir.IntegerAttr(softmax.attributes["dimension"]).value)
            stable_attr_name = (
                "numericStable"
                if "numericStable" in softmax.attributes
                else "numeric_stable"
            )
            numeric_stable = bool(
                ir.BoolAttr(softmax.attributes[stable_attr_name]).value
            )
        except (KeyError, TypeError, ValueError):
            return None
        rank = len(_shape(softmax.operands[0]))
        if dimension not in (-1, rank - 1) or not numeric_stable:
            return None
        return softmax.operands[0]

    probabilities = _producer(value, "ttir.div")
    if probabilities is None or len(probabilities.operands) != 2:
        return None
    numerator_value = _strip_attention_views(probabilities.operands[0])
    if numerator_value is None:
        return None
    numerator = _producer(numerator_value, "ttir.exp")
    denominator_value = _strip_attention_views(probabilities.operands[1])
    if denominator_value is None:
        return None
    denominator = _producer(denominator_value, "ttir.sum")
    if numerator is None or denominator is None:
        return None
    denominator_input = _strip_attention_views(denominator.operands[0])
    if denominator_input is None or denominator_input != numerator.results[0]:
        return None
    if not _reduction_is_last_dim(denominator, numerator.results[0]):
        return None

    numerator_input = _strip_attention_views(numerator.operands[0])
    if numerator_input is None:
        return None
    shifted = _producer(numerator_input, "ttir.subtract")
    if shifted is None or len(shifted.operands) != 2:
        return None
    scores = shifted.operands[0]
    row_max_value = _strip_attention_views(shifted.operands[1])
    if row_max_value is None:
        return None
    row_max = _producer(row_max_value, "ttir.max")
    if row_max is None:
        return None
    row_max_input = _strip_attention_views(row_max.operands[0])
    if row_max_input is None or row_max_input != scores:
        return None
    if not _reduction_is_last_dim(row_max, scores):
        return None
    return scores


def _valid_score_matmul(op, key_transposed):
    if op.name == "ttir.matmul":
        transpose_b = False
        if "transpose_b" in op.attributes:
            transpose_b = bool(ir.BoolAttr(op.attributes["transpose_b"]).value)
        return transpose_b != key_transposed
    if op.name != "ttir.dot_general":
        return False
    try:
        lhs_dims = tuple(int(value) for value in op.attributes["contract_dims_lhs"])
        rhs_dims = tuple(int(value) for value in op.attributes["contract_dims_rhs"])
        lhs_rank = len(_shape(op.operands[0]))
        rhs_rank = len(_shape(op.operands[1]))
    except (KeyError, TypeError, ValueError):
        return False
    expected_rhs = rhs_rank - 2 if key_transposed else rhs_rank - 1
    return lhs_dims == (lhs_rank - 1,) and rhs_dims == (expected_rhs,)


def _match_score_matmul(value):
    score_path = _analyze_attention_operand(value)
    if score_path is None:
        return None
    qk = _producer(score_path.source, ("ttir.matmul", "ttir.dot_general"))
    if qk is None or len(qk.operands) != 2:
        return None

    query_path = _analyze_attention_operand(qk.operands[0])
    key_path = _analyze_attention_operand(qk.operands[1], allow_transpose=True)
    if query_path is None or key_path is None:
        return None
    if query_path.transposed or not _valid_score_matmul(qk, key_path.transposed):
        return None
    return query_path, key_path, score_path


def _match_decomposed_prefill_sdpa(op):
    """Recognize matmul(stable_softmax(QK * scale + mask), V)."""

    try:
        if len(op.operands) != 1 or len(op.results) != 1 or not _is_bf16(op.results[0]):
            return None

        output_value = _strip_attention_views(op.operands[0])
        if output_value is None:
            return None
        output_matmul = _producer(output_value, ("ttir.matmul", "ttir.dot_general"))
        if output_matmul is None or len(output_matmul.operands) != 2:
            return None

        scores = _match_softmax_input(output_matmul.operands[0])
        scores_with_mask = _producer(scores, "ttir.add")
        if scores_with_mask is None or len(scores_with_mask.operands) != 2:
            return None

        score_match = None
        mask_branch = None
        for operand in scores_with_mask.operands:
            candidate = _match_score_matmul(operand)
            if candidate is not None:
                if score_match is not None:
                    return None
                score_match = candidate
            else:
                mask_branch = operand
        if score_match is None or mask_branch is None:
            return None

        value_path = _analyze_attention_operand(output_matmul.operands[1])
        mask_path = _analyze_attention_operand(mask_branch)
        if value_path is None or mask_path is None:
            return None
        if value_path.scale != 1.0 or value_path.transposed:
            return None
        query_path, key_path, score_path = score_match
        query = query_path.source
        key = key_path.source
        value = value_path.source
        mask = mask_path.source
        scale = score_path.scale * query_path.scale * key_path.scale
        match = _make_prefill_sdpa(query, key, value, mask, scale, is_causal=False)
        if match is None:
            return None
        if _shape(op.results[0]) != match.query_shape:
            return None
        expected_scores = (
            match.query_shape[0],
            match.query_shape[1],
            match.query_shape[2],
            match.key_shape[2],
        )
        expanded_key_shape = (
            match.query_shape[0],
            match.query_shape[1],
            match.key_shape[2],
            match.key_shape[3],
        )
        if (
            _shape(scores_with_mask.results[0]) != expected_scores
            or not _valid_score_path(score_path, expected_scores)
            or not _valid_attention_path(
                query_path, match.query_shape, match.query_shape
            )
            or not _valid_attention_path(
                key_path,
                match.key_shape,
                expanded_key_shape,
                transpose=key_path.transposed,
            )
            or not _valid_attention_path(
                value_path, match.key_shape, expanded_key_shape
            )
            or not _valid_mask_path(mask_path, expected_scores)
        ):
            return None
        return match
    except (IndexError, KeyError, TypeError, ValueError):
        return None


@d2m.pattern(
    root="ttir.typecast",
    benefit=300,
    match=lambda op: _match_decomposed_prefill_sdpa(op) is not None,
)
def lower_decomposed_prefill_sdpa(op, rewriter):
    return _lower_prefill_sdpa(_match_decomposed_prefill_sdpa(op))


def _llama_prefill_sdpa_golden(q, k, v):
    query_heads_per_kv = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(query_heads_per_kv, dim=1)
    v = v.repeat_interleave(query_heads_per_kv, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    causal_mask = torch.triu(
        torch.full((q.shape[-2], k.shape[-2]), float("-inf"), dtype=scores.dtype),
        diagonal=1,
    )
    probabilities = torch.softmax(scores + causal_mask, dim=-1)
    return torch.matmul(probabilities, v)


def _masked_prefill_sdpa_golden(q, k, v, mask):
    query_heads_per_kv = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(query_heads_per_kv, dim=1)
    v = v.repeat_interleave(query_heads_per_kv, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * 0.125
    return torch.matmul(torch.softmax(scores + mask, dim=-1), v)


def _cache_sdpa_inputs(shape, dtype, generator):
    if tuple(shape) == (_BATCH, 1, _SEQUENCE, _CACHE_SEQUENCE):
        mask = torch.full(shape, float("-inf"), dtype=dtype)
        mask[:, :, :, :_SEQUENCE] = torch.triu(
            torch.full((_SEQUENCE, _SEQUENCE), float("-inf"), dtype=dtype),
            diagonal=1,
        )
        return mask
    return (
        torch.rand(shape, generator=generator, dtype=torch.float32).to(dtype) * 0.5
        - 0.25
    )


def _llama_prefill_fill_cache_golden(cache, input):
    result = cache.clone()
    result[:, :, : input.shape[2], :] = input
    return result


def _llama_prefill_cache_sdpa_golden(
    q, key_cache, key_update, value_cache, value_update, mask
):
    key_cache = key_cache.clone()
    value_cache = value_cache.clone()
    key_cache[:, :, :_SEQUENCE, :] = key_update
    value_cache[:, :, :_SEQUENCE, :] = value_update

    query_heads_per_kv = q.shape[1] // key_cache.shape[1]
    key = key_cache.repeat_interleave(query_heads_per_kv, dim=1)
    value = value_cache.repeat_interleave(query_heads_per_kv, dim=1)
    scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    probabilities = torch.softmax(scores + mask, dim=-1)
    return torch.matmul(probabilities, value)


def _llama_prefill_fill_cache_chain_ttir(sequence=_SEQUENCE):
    operations = []
    previous = "%cache"
    for batch_offset in range(_BATCH):
        slice_result = f"%slice{batch_offset}"
        update_result = f"%updated{batch_offset}"
        operations.append(
            f"""
            {slice_result} = "ttir.slice_static"(%input) <{{
              begins = [{batch_offset} : i32, 0 : i32, 0 : i32, 0 : i32],
              ends = [{batch_offset + 1} : i32, 8 : i32, {sequence} : i32, 128 : i32],
              step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
            }}> : (tensor<32x8x{sequence}x128xbf16>) -> tensor<1x8x{sequence}x128xbf16>
            {update_result} = "ttir.fill_cache"({previous}, {slice_result}) <{{
              batch_offset = {batch_offset} : i32
            }}> : (tensor<32x8x128x128xbf16>, tensor<1x8x{sequence}x128xbf16>) -> tensor<32x8x128x128xbf16>"""
        )
        previous = update_result

    body = "".join(operations)
    return f"""
    module {{
      func.func @forward(
          %cache: tensor<32x8x128x128xbf16>,
          %input: tensor<32x8x{sequence}x128xbf16>) -> tensor<32x8x128x128xbf16> {{
        {body}
        return {previous} : tensor<32x8x128x128xbf16>
      }}
    }}
    """


def _native_softmax_gqa_ttir(canonical_key_expansion=True):
    if canonical_key_expansion:
        key_unit_shape = "1x2x1x64x32"
        key_grouped_shape = "1x2x2x64x32"
        key_broadcast = "1, 1, 2, 1, 1"
    else:
        key_unit_shape = "1x2x64x1x32"
        key_grouped_shape = "1x2x64x2x32"
        key_broadcast = "1, 1, 1, 2, 1"

    return f"""
    module {{
      func.func @forward(
          %q: tensor<1x4x32x32xbf16>,
          %k: tensor<1x2x64x32xbf16>,
          %v: tensor<1x2x64x32xbf16>,
          %mask: tensor<1x1x32x64xbf16>) -> tensor<1x4x32x32xbf16> {{
        %qf = "ttir.typecast"(%q) <{{conservative_folding = false}}> :
          (tensor<1x4x32x32xbf16>) -> tensor<1x4x32x32xf32>
        %k0 = "ttir.reshape"(%k) <{{shape = [{", ".join(f"{dim} : i32" for dim in key_unit_shape.split("x"))}]}}> :
          (tensor<1x2x64x32xbf16>) -> tensor<{key_unit_shape}xbf16>
        %k1 = "ttir.broadcast"(%k0) <{{broadcast_dimensions = array<i64: {key_broadcast}>}}> :
          (tensor<{key_unit_shape}xbf16>) -> tensor<{key_grouped_shape}xbf16>
        %k2 = "ttir.reshape"(%k1) <{{shape = [1 : i32, 4 : i32, 64 : i32, 32 : i32]}}> :
          (tensor<{key_grouped_shape}xbf16>) -> tensor<1x4x64x32xbf16>
        %kf = "ttir.typecast"(%k2) <{{conservative_folding = false}}> :
          (tensor<1x4x64x32xbf16>) -> tensor<1x4x64x32xf32>
        %kt = "ttir.permute"(%kf) <{{permutation = array<i64: 0, 1, 3, 2>}}> :
          (tensor<1x4x64x32xf32>) -> tensor<1x4x32x64xf32>
        %qk = "ttir.matmul"(%qf, %kt) <{{transpose_a = false, transpose_b = false}}> :
          (tensor<1x4x32x32xf32>, tensor<1x4x32x64xf32>) -> tensor<1x4x32x64xf32>
        %scale = "ttir.full"() <{{
          fill_value = 0.125 : f32,
          shape = array<i32: 1, 4, 32, 64>
        }}> : () -> tensor<1x4x32x64xf32>
        %scaled = "ttir.multiply"(%qk, %scale) :
          (tensor<1x4x32x64xf32>, tensor<1x4x32x64xf32>) -> tensor<1x4x32x64xf32>
        %maskf = "ttir.typecast"(%mask) <{{conservative_folding = false}}> :
          (tensor<1x1x32x64xbf16>) -> tensor<1x1x32x64xf32>
        %masked = "ttir.add"(%scaled, %maskf) :
          (tensor<1x4x32x64xf32>, tensor<1x1x32x64xf32>) -> tensor<1x4x32x64xf32>
        %probabilities = "ttir.softmax"(%masked) <{{
          dimension = 3 : si32,
          numericStable = true
        }}> : (tensor<1x4x32x64xf32>) -> tensor<1x4x32x64xf32>
        %v0 = "ttir.reshape"(%v) <{{shape = [1 : i32, 2 : i32, 1 : i32, 64 : i32, 32 : i32]}}> :
          (tensor<1x2x64x32xbf16>) -> tensor<1x2x1x64x32xbf16>
        %v1 = "ttir.broadcast"(%v0) <{{broadcast_dimensions = array<i64: 1, 1, 2, 1, 1>}}> :
          (tensor<1x2x1x64x32xbf16>) -> tensor<1x2x2x64x32xbf16>
        %v2 = "ttir.reshape"(%v1) <{{shape = [1 : i32, 4 : i32, 64 : i32, 32 : i32]}}> :
          (tensor<1x2x2x64x32xbf16>) -> tensor<1x4x64x32xbf16>
        %vf = "ttir.typecast"(%v2) <{{conservative_folding = false}}> :
          (tensor<1x4x64x32xbf16>) -> tensor<1x4x64x32xf32>
        %outf = "ttir.matmul"(%probabilities, %vf) <{{transpose_a = false, transpose_b = false}}> :
          (tensor<1x4x32x64xf32>, tensor<1x4x64x32xf32>) -> tensor<1x4x32x32xf32>
        %out = "ttir.typecast"(%outf) <{{conservative_folding = false}}> :
          (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xbf16>
        return %out : tensor<1x4x32x32xbf16>
      }}
    }}
    """


PATTERN_TESTS = [
    PatternTest(
        name="llama_prefill_cache_sdpa_lowers",
        ttir="""
        module {
          func.func @forward(
              %q: tensor<32x32x18x128xbf16>,
              %key_cache: tensor<32x8x128x128xbf16>,
              %key_update: tensor<32x8x18x128xbf16>,
              %value_cache: tensor<32x8x128x128xbf16>,
              %value_update: tensor<32x8x18x128xbf16>,
              %mask: tensor<32x1x18x128xbf16>) -> tensor<32x32x18x128xbf16> {
            %key = "ttir.fill_cache"(%key_cache, %key_update) <{
              batch_offset = 0 : i32
            }> : (tensor<32x8x128x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
            %value = "ttir.fill_cache"(%value_cache, %value_update) <{
              batch_offset = 0 : i32
            }> : (tensor<32x8x128x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>

            %qf = "ttir.typecast"(%q) <{conservative_folding = false}> :
              (tensor<32x32x18x128xbf16>) -> tensor<32x32x18x128xf32>
            %qscale = "ttir.full"() <{
              fill_value = 0.297301769 : f32,
              shape = array<i32: 32, 32, 18, 128>
            }> : () -> tensor<32x32x18x128xf32>
            %qs = "ttir.multiply"(%qf, %qscale) :
              (tensor<32x32x18x128xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>

            %k0 = "ttir.reshape"(%key) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 128 : i32]}> :
              (tensor<32x8x128x128xbf16>) -> tensor<32x8x1x128x128xbf16>
            %k1 = "ttir.typecast"(%k0) <{conservative_folding = false}> :
              (tensor<32x8x1x128x128xbf16>) -> tensor<32x8x1x128x128xf32>
            %kscale = "ttir.full"() <{
              fill_value = 0.297301769 : f32,
              shape = array<i32: 32, 8, 4, 128, 128>
            }> : () -> tensor<32x8x4x128x128xf32>
            %k2 = "ttir.multiply"(%k1, %kscale) :
              (tensor<32x8x1x128x128xf32>, tensor<32x8x4x128x128xf32>) -> tensor<32x8x4x128x128xf32>
            %k3 = "ttir.reshape"(%k2) <{shape = [32 : i32, 32 : i32, 128 : i32, 128 : i32]}> :
              (tensor<32x8x4x128x128xf32>) -> tensor<32x32x128x128xf32>
            %kt = "ttir.permute"(%k3) <{permutation = array<i64: 0, 1, 3, 2>}> :
              (tensor<32x32x128x128xf32>) -> tensor<32x32x128x128xf32>
            %scores = "ttir.matmul"(%qs, %kt) <{transpose_a = false, transpose_b = false}> :
              (tensor<32x32x18x128xf32>, tensor<32x32x128x128xf32>) -> tensor<32x32x18x128xf32>

            %maskf = "ttir.typecast"(%mask) <{conservative_folding = false}> :
              (tensor<32x1x18x128xbf16>) -> tensor<32x1x18x128xf32>
            %masked = "ttir.add"(%scores, %maskf) :
              (tensor<32x32x18x128xf32>, tensor<32x1x18x128xf32>) -> tensor<32x32x18x128xf32>
            %max = "ttir.max"(%masked) <{dim_arg = [3 : i32], keep_dim = false}> :
              (tensor<32x32x18x128xf32>) -> tensor<32x32x18xf32>
            %max4d = "ttir.reshape"(%max) <{shape = [32 : i32, 32 : i32, 18 : i32, 1 : i32]}> :
              (tensor<32x32x18xf32>) -> tensor<32x32x18x1xf32>
            %shifted = "ttir.subtract"(%masked, %max4d) :
              (tensor<32x32x18x128xf32>, tensor<32x32x18x1xf32>) -> tensor<32x32x18x128xf32>
            %numerator = "ttir.exp"(%shifted) :
              (tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
            %sum = "ttir.sum"(%numerator) <{dim_arg = [3 : i32], keep_dim = false}> :
              (tensor<32x32x18x128xf32>) -> tensor<32x32x18xf32>
            %sum4d = "ttir.reshape"(%sum) <{shape = [32 : i32, 32 : i32, 18 : i32, 1 : i32]}> :
              (tensor<32x32x18xf32>) -> tensor<32x32x18x1xf32>
            %probabilities = "ttir.div"(%numerator, %sum4d) :
              (tensor<32x32x18x128xf32>, tensor<32x32x18x1xf32>) -> tensor<32x32x18x128xf32>
            %condition = "ttir.zeros"() <{shape = array<i32: 32, 32, 18, 1>}> :
              () -> tensor<32x32x18x1xf32>
            %zero = "ttir.zeros"() <{shape = array<i32: 32, 32, 18, 128>}> :
              () -> tensor<32x32x18x128xf32>
            %softmax = "ttir.where"(%condition, %zero, %probabilities) :
              (tensor<32x32x18x1xf32>, tensor<32x32x18x128xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>

            %v0 = "ttir.reshape"(%value) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 128 : i32]}> :
              (tensor<32x8x128x128xbf16>) -> tensor<32x8x1x128x128xbf16>
            %v1 = "ttir.typecast"(%v0) <{conservative_folding = false}> :
              (tensor<32x8x1x128x128xbf16>) -> tensor<32x8x1x128x128xf32>
            %v2 = "ttir.broadcast"(%v1) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> :
              (tensor<32x8x1x128x128xf32>) -> tensor<32x8x4x128x128xf32>
            %v3 = "ttir.reshape"(%v2) <{shape = [32 : i32, 32 : i32, 128 : i32, 128 : i32]}> :
              (tensor<32x8x4x128x128xf32>) -> tensor<32x32x128x128xf32>
            %outf = "ttir.matmul"(%softmax, %v3) <{transpose_a = false, transpose_b = false}> :
              (tensor<32x32x18x128xf32>, tensor<32x32x128x128xf32>) -> tensor<32x32x18x128xf32>
            %out = "ttir.typecast"(%outf) <{conservative_folding = false}> :
              (tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xbf16>
            return %out : tensor<32x32x18x128xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.matmul
        CHECK: arith.constant dense<8.837890e-02>
        CHECK-COUNT-5: d2m.generic
        CHECK: grid = #ttcore.grid<8x4>
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_reduce_max
        CHECK: d2m.tile_reduce_sum
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_bcast
        CHECK: d2m.tile_div
        CHECK: d2m.independent_loop
        """,
        golden=_llama_prefill_cache_sdpa_golden,
        inputs=InputSpec(_cache_sdpa_inputs, seed=0),
        pcc=0.98,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="llama_prefill_fill_cache_chain_s16_lowers",
        ttir=_llama_prefill_fill_cache_chain_ttir(sequence=16),
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.slice_static
        CHECK-NOT: ttir.fill_cache
        CHECK: d2m.generic
        CHECK: d2m.tile_where
        CHECK: d2m.remote_store
        CHECK: d2m.generic
        CHECK: d2m.remote_store
        CHECK: d2m.to_layout %{{.*}}#1
        """,
        golden=_llama_prefill_fill_cache_golden,
        inputs=InputSpec("uniform(-0.25,0.25)", seed=0),
        pcc=0.999,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="llama_prefill_fill_cache_lowers",
        ttir="""
        module {
          func.func @forward(
              %cache: tensor<32x8x128x128xbf16>,
              %input: tensor<32x8x18x128xbf16>) -> tensor<32x8x128x128xbf16> {
            %0 = "ttir.fill_cache"(%cache, %input) <{
              batch_offset = 0 : i32
            }> : (tensor<32x8x128x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
            return %0 : tensor<32x8x128x128xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.fill_cache
        CHECK: d2m.generic
        CHECK: d2m.tile_where
        CHECK: d2m.remote_store
        """,
        golden=_llama_prefill_fill_cache_golden,
        inputs=InputSpec("uniform(-0.25,0.25)", seed=0),
        pcc=0.999,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="llama_prefill_sdpa_lowers",
        ttir="""
        module {
          func.func @forward(
              %q: tensor<32x32x18x128xbf16>,
              %k: tensor<32x8x18x128xbf16>,
              %v: tensor<32x8x18x128xbf16>) -> tensor<32x32x18x128xbf16> {
            %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v) <{
              is_causal = true,
              operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>
            }> : (tensor<32x32x18x128xbf16>, tensor<32x8x18x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x32x18x128xbf16>
            return %0 : tensor<32x32x18x128xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.scaled_dot_product_attention
        CHECK-COUNT-1: d2m.generic
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_reduce_max
        CHECK: d2m.tile_exp
        CHECK: d2m.tile_reduce_sum
        CHECK: d2m.tile_matmul
        CHECK: d2m.remote_store
        CHECK: } {d2m.independent_loop}
        """,
        golden=_llama_prefill_sdpa_golden,
        inputs=InputSpec("uniform(-0.25,0.25)", seed=0),
        pcc=0.98,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="prefill_sdpa_multitile_gqa_lowers",
        ttir="""
        module {
          func.func @forward(
              %q: tensor<2x4x64x64xbf16>,
              %k: tensor<2x2x64x64xbf16>,
              %v: tensor<2x2x64x64xbf16>) -> tensor<2x4x64x64xbf16> {
            %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v) <{
              is_causal = true,
              operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>
            }> : (tensor<2x4x64x64xbf16>, tensor<2x2x64x64xbf16>, tensor<2x2x64x64xbf16>) -> tensor<2x4x64x64xbf16>
            return %0 : tensor<2x4x64x64xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.scaled_dot_product_attention
        CHECK-COUNT-1: d2m.generic
        CHECK: grid = #ttcore.grid<8x2>
        CHECK: scf.for
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_reduce_max
        CHECK: d2m.tile_exp
        CHECK: d2m.tile_reduce_sum
        CHECK: d2m.tile_bcast
        CHECK: d2m.tile_div
        CHECK: d2m.remote_store
        CHECK: } {d2m.independent_loop}
        """,
        golden=_llama_prefill_sdpa_golden,
        inputs=InputSpec("uniform(-0.25,0.25)", seed=0),
        pcc=0.98,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="prefill_sdpa_nonsquare_head_mask_lowers",
        ttir="""
        module {
          func.func @forward(
              %q: tensor<2x4x64x64xbf16>,
              %k: tensor<2x2x80x64xbf16>,
              %v: tensor<2x2x80x64xbf16>,
              %mask: tensor<1x4x64x80xbf16>) -> tensor<2x4x64x64xbf16> {
            %0 = "ttir.scaled_dot_product_attention"(%q, %k, %v, %mask) <{
              is_causal = false,
              operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
              scale = 0.125 : f32
            }> : (tensor<2x4x64x64xbf16>, tensor<2x2x80x64xbf16>, tensor<2x2x80x64xbf16>, tensor<1x4x64x80xbf16>) -> tensor<2x4x64x64xbf16>
            return %0 : tensor<2x4x64x64xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.scaled_dot_product_attention
        CHECK-COUNT-1: d2m.generic
        CHECK: grid = #ttcore.grid<4x2>
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_reduce_max
        CHECK: d2m.tile_exp
        CHECK: d2m.tile_reduce_sum
        CHECK: d2m.tile_bcast
        CHECK: d2m.tile_div
        CHECK: d2m.remote_store
        """,
        golden=_masked_prefill_sdpa_golden,
        inputs=InputSpec("uniform(-0.25,0.25)", seed=0),
        pcc=0.98,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="decomposed_prefill_sdpa_nonsquare_gqa_lowers",
        ttir="""
        module {
          func.func @forward(
              %q: tensor<2x4x64x64xbf16>,
              %k: tensor<2x2x96x64xbf16>,
              %v: tensor<2x2x96x64xbf16>,
              %mask: tensor<1x4x64x96xbf16>) -> tensor<2x4x64x64xbf16> {
            %qf = "ttir.typecast"(%q) <{conservative_folding = false}> :
              (tensor<2x4x64x64xbf16>) -> tensor<2x4x64x64xf32>

            %k0 = "ttir.reshape"(%k) <{shape = [2 : i32, 2 : i32, 1 : i32, 96 : i32, 64 : i32]}> :
              (tensor<2x2x96x64xbf16>) -> tensor<2x2x1x96x64xbf16>
            %k1 = "ttir.broadcast"(%k0) <{broadcast_dimensions = array<i64: 1, 1, 2, 1, 1>}> :
              (tensor<2x2x1x96x64xbf16>) -> tensor<2x2x2x96x64xbf16>
            %k2 = "ttir.reshape"(%k1) <{shape = [2 : i32, 4 : i32, 96 : i32, 64 : i32]}> :
              (tensor<2x2x2x96x64xbf16>) -> tensor<2x4x96x64xbf16>
            %kf = "ttir.typecast"(%k2) <{conservative_folding = false}> :
              (tensor<2x4x96x64xbf16>) -> tensor<2x4x96x64xf32>
            %kt = "ttir.permute"(%kf) <{permutation = array<i64: 0, 1, 3, 2>}> :
              (tensor<2x4x96x64xf32>) -> tensor<2x4x64x96xf32>
            %qk = "ttir.matmul"(%qf, %kt) <{transpose_a = false, transpose_b = false}> :
              (tensor<2x4x64x64xf32>, tensor<2x4x64x96xf32>) -> tensor<2x4x64x96xf32>
            %scale = "ttir.full"() <{
              fill_value = 0.125 : f32,
              shape = array<i32: 2, 4, 64, 96>
            }> : () -> tensor<2x4x64x96xf32>
            %scaled = "ttir.multiply"(%qk, %scale) :
              (tensor<2x4x64x96xf32>, tensor<2x4x64x96xf32>) -> tensor<2x4x64x96xf32>
            %maskf = "ttir.typecast"(%mask) <{conservative_folding = false}> :
              (tensor<1x4x64x96xbf16>) -> tensor<1x4x64x96xf32>
            %masked = "ttir.add"(%scaled, %maskf) :
              (tensor<2x4x64x96xf32>, tensor<1x4x64x96xf32>) -> tensor<2x4x64x96xf32>
            %max = "ttir.max"(%masked) <{dim_arg = [3 : i32], keep_dim = false}> :
              (tensor<2x4x64x96xf32>) -> tensor<2x4x64xf32>
            %max4d = "ttir.reshape"(%max) <{shape = [2 : i32, 4 : i32, 64 : i32, 1 : i32]}> :
              (tensor<2x4x64xf32>) -> tensor<2x4x64x1xf32>
            %shifted = "ttir.subtract"(%masked, %max4d) :
              (tensor<2x4x64x96xf32>, tensor<2x4x64x1xf32>) -> tensor<2x4x64x96xf32>
            %numerator = "ttir.exp"(%shifted) :
              (tensor<2x4x64x96xf32>) -> tensor<2x4x64x96xf32>
            %sum = "ttir.sum"(%numerator) <{dim_arg = [3 : i32], keep_dim = false}> :
              (tensor<2x4x64x96xf32>) -> tensor<2x4x64xf32>
            %sum4d = "ttir.reshape"(%sum) <{shape = [2 : i32, 4 : i32, 64 : i32, 1 : i32]}> :
              (tensor<2x4x64xf32>) -> tensor<2x4x64x1xf32>
            %probabilities = "ttir.div"(%numerator, %sum4d) :
              (tensor<2x4x64x96xf32>, tensor<2x4x64x1xf32>) -> tensor<2x4x64x96xf32>

            %v0 = "ttir.reshape"(%v) <{shape = [2 : i32, 2 : i32, 1 : i32, 96 : i32, 64 : i32]}> :
              (tensor<2x2x96x64xbf16>) -> tensor<2x2x1x96x64xbf16>
            %v1 = "ttir.broadcast"(%v0) <{broadcast_dimensions = array<i64: 1, 1, 2, 1, 1>}> :
              (tensor<2x2x1x96x64xbf16>) -> tensor<2x2x2x96x64xbf16>
            %v2 = "ttir.reshape"(%v1) <{shape = [2 : i32, 4 : i32, 96 : i32, 64 : i32]}> :
              (tensor<2x2x2x96x64xbf16>) -> tensor<2x4x96x64xbf16>
            %vf = "ttir.typecast"(%v2) <{conservative_folding = false}> :
              (tensor<2x4x96x64xbf16>) -> tensor<2x4x96x64xf32>
            %outf = "ttir.matmul"(%probabilities, %vf) <{transpose_a = false, transpose_b = false}> :
              (tensor<2x4x64x96xf32>, tensor<2x4x96x64xf32>) -> tensor<2x4x64x64xf32>
            %out = "ttir.typecast"(%outf) <{conservative_folding = false}> :
              (tensor<2x4x64x64xf32>) -> tensor<2x4x64x64xbf16>
            return %out : tensor<2x4x64x64xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.matmul
        CHECK-NOT: ttir.exp
        CHECK: arith.constant dense<1.250000e-01>
        CHECK-COUNT-1: d2m.generic
        CHECK: grid = #ttcore.grid<4x2>
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_reduce_max
        CHECK: d2m.tile_reduce_sum
        CHECK: d2m.tile_matmul
        CHECK: d2m.tile_div
        """,
        golden=_masked_prefill_sdpa_golden,
        inputs=InputSpec("uniform(-0.25,0.25)", seed=0),
        pcc=0.98,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="decomposed_native_softmax_gqa_lowers",
        ttir=_native_softmax_gqa_ttir(),
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.softmax
        CHECK-NOT: ttir.matmul
        CHECK-COUNT-1: d2m.generic
        CHECK: d2m.tile_reduce_max
        CHECK: d2m.tile_reduce_sum
        CHECK: d2m.tile_div
        """,
        golden=_masked_prefill_sdpa_golden,
        inputs=InputSpec("uniform(-0.25,0.25)", seed=0),
        pcc=0.98,
        use_tile_matmul=False,
    ),
    PatternTest(
        name="decomposed_sdpa_rejects_noncanonical_gqa_expansion",
        ttir=_native_softmax_gqa_ttir(canonical_key_expansion=False),
        check="""
        CHECK-LABEL: func.func @forward
        CHECK: ttir.matmul
        CHECK: ttir.softmax
        CHECK-NOT: d2m.generic
        """,
        expect_match=False,
    ),
]
