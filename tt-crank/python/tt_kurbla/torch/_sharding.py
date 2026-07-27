"""DTensor op sharding strategies the tt backend needs but torch lacks.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding


def _sdpa_overrideable_sharding(
    query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False, return_debug_mask=False, scale=None
):
    """
    Batch/head-parallel DTensor sharding for the overrideable SDPA op tt selects.

    Torch ships no sharding strategy for the overrideable op, so without this DTensor
    fails propagation ("does not have a sharding strategy registered").

    Attention is independent across both batch (dim 0) and head (dim 1): Q/K/V sharded
    on either flow straight to output/logsumexp on the same dim with no gather. Three
    combos are offered (all-replicate as the fallback, batch `Shard(0)` for DP, head
    `Shard(1)` for TP), so DP and TP each match their inputs at zero redistribution
    cost; without the batch combo DP's `Shard(0)` inputs match nothing and DTensor must
    redistribute them (a collective), losing the batch-parallel path. The other 7
    outputs (cum_seq/max/philox/debug) are empty or per-mesh scalars -> `None`/`Replicate`.
    The mask is 4-D `[B, H, S, S]` (asserted), usually broadcast (size 1) along batch
    and head so it stays `Replicate`; when full along the sharded dim it shards on that
    dim too (batch in the batch combo, head in the head combo) so its per-shard extent
    matches the query (the op verifier requires mask batch/head == 1 or == query's).

    Returns (output_placements, input_placements) combos. Input placements cover the
    tensor args (query, key, value, [attn_bias]); the trailing scalar args
    (dropout_p, is_causal, return_debug_mask, scale) are omitted.
    """
    has_bias = attn_bias is not None
    if has_bias:
        assert attn_bias.ndim == 4, (
            f"attn_mask must be 4-D [B, H, S, S] for the DTensor SDPA strategy, got rank {attn_bias.ndim}"
        )
    # Output 8 (debug_attn_mask) is left None, valid only when no debug mask is produced.
    assert not return_debug_mask, "DTensor SDPA strategy does not support return_debug_mask=True"
    # 9 outputs: output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask.
    out_replicate = [Replicate(), Replicate(), None, None, None, None, Replicate(), None, None]
    out_batch = [Shard(0), Shard(0), None, None, None, None, Replicate(), None, None]
    out_head = [Shard(1), Shard(1), None, None, None, None, Replicate(), None, None]
    bias = [Replicate()] if has_bias else []
    # Shard the mask on a dim only when it is full there (size != 1); a broadcast
    # (size-1) mask stays Replicate and applies on every shard unchanged.
    batch_bias = [Shard(0)] if (has_bias and attn_bias.shape[0] != 1) else bias
    head_bias = [Shard(1)] if (has_bias and attn_bias.shape[1] != 1) else bias
    return [
        (out_replicate, [Replicate(), Replicate(), Replicate()] + bias),
        (out_batch, [Shard(0), Shard(0), Shard(0)] + batch_bias),
        (out_head, [Shard(1), Shard(1), Shard(1)] + head_bias),
    ]


def _index_copy_sharding(self, dim, index, source):
    """DTensor sharding strategy for `index_copy_`/`index_copy`.

    The write is along `dim` with a replicated 1-D index; self and source may
    be sharded on any *other* dim and the output keeps that placement. The index
    is replicated in every case - it addresses the (unsharded) `dim`
    identically on every shard. Returns (output_placements, per-positional-arg
    input_placements); non-tensor args (`dim`) take `None`.
    """
    d = dim if dim >= 0 else dim + self.ndim
    shardings = [([Replicate()], [Replicate(), None, Replicate(), Replicate()])]
    for sd in range(self.ndim):
        if sd != d:
            shardings.append(([Shard(sd)], [Shard(sd), None, Replicate(), Shard(sd)]))
    return shardings


def register_sharding_strategies() -> None:
    """Register the tt-specific DTensor sharding strategies on the propagator.

    All go through the public `register_sharding`: the overrideable SDPA op (which
    torch ships a strategy for only in its CUDA-family fused variants) and both the
    in-place and functional `index_copy` overloads (which appear depending on
    whether the write is traced (compile) or run eagerly).
    """
    aten = torch.ops.aten
    register_sharding(aten._scaled_dot_product_fused_attention_overrideable.default)(_sdpa_overrideable_sharding)
    register_sharding(aten.index_copy_.default)(_index_copy_sharding)
    register_sharding(aten.index_copy.default)(_index_copy_sharding)
