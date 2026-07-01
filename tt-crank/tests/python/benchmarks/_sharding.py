"""DTensor sharding helpers for the multi-chip Llama benchmark.

Two parallelism strategies, both eager DTensor over a 1-D chip mesh:

- Data parallel (DP): model replicated, the batch sharded across chips.
- Tensor parallel (TP, Megatron): each decoder layer's attention/MLP linears
  are column/row sharded so the per-token compute is split across chips.

The KV cache is a `StaticCache`, sharded to match the model: along batch
(dim 0) for DP, along the kv-head dim (dim 1) for TP. `StaticCacheLayer.update`
writes new tokens with an in-place `index_copy_` along the sequence dim
(dim 2) - orthogonal to both shard dims, so the write never crosses a shard
boundary. This mirrors tt-xla's StaticCache + `mark_sharding` setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)

# Megatron column-parallel weights (shard the output/feature dim) and
# row-parallel weights (shard the contraction dim). lm_head / embed_tokens are
# left replicated: Llama-3.2-1B ties them (one weight), so vocab-sharding lm_head
# would also vocab-shard the embedding lookup and corrupt it.
_COLUMN_PARALLEL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
_ROW_PARALLEL = ("o_proj", "down_proj")


def tensor_parallel(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Megatron tensor-parallel: column/row shard each decoder layer's
    attention and MLP projections; everything else replicated."""

    def partition_fn(name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        if not isinstance(module, nn.Linear):
            return
        leaf = name.rsplit(".", 1)[-1]
        if leaf in _COLUMN_PARALLEL:
            placements = [Shard(0)]
        elif leaf in _ROW_PARALLEL:
            placements = [Shard(1)]
        else:
            return
        w = module.weight
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(w, device_mesh, placements),
                requires_grad=w.requires_grad,
            ),
        )

    return distribute_module(model.to("tt"), mesh, partition_fn=partition_fn)


def shard_static_cache(cache, mesh: DeviceMesh, *, parallel: str) -> None:
    """Replace each cache layer's k/v tensors with DTensors in-place.

    DP shards the batch dim (0); TP shards the kv-head dim (1). The cache write
    (`index_copy_` along dim 2) stays within a shard.

    Also swaps each layer's `reset` for one that redistributes fresh zeros:
    `reset` otherwise zeroes the k/v in place with `zero_()`, which on a
    sharded tensor hits the same eager host-roundtrip write that has no path
    here. `run_llm_benchmark` resets between warmup and the timed loop.
    """
    cache_dim = 0 if parallel == "dp" else 1
    placements = [Shard(cache_dim)]
    for layer in cache.layers:
        layer.keys = distribute_tensor(layer.keys, mesh, placements)
        layer.values = distribute_tensor(layer.values, mesh, placements)
        _install_sharded_reset(layer, mesh, placements)


def _install_sharded_reset(layer, mesh: DeviceMesh, placements) -> None:
    """Swap `layer.reset` for one that rebuilds the k/v as fresh sharded zeros -
    HF's in-place `zero_()` can't write back into a sharded tensor."""

    def reset() -> None:
        for attr in ("keys", "values"):
            t = getattr(layer, attr)
            zeros = torch.zeros(tuple(t.shape), dtype=t.dtype).to("tt")
            setattr(layer, attr, distribute_tensor(zeros, mesh, placements))
        cum = getattr(layer, "cumulative_length", None)
        if isinstance(cum, int):
            layer.cumulative_length = 0
        elif cum is not None:
            cum.zero_()  # small per-layer counter, not sharded

    layer.reset = reset


def distribute_inputs(
    input_ids: torch.Tensor,
    cache_position: torch.Tensor,
    mesh: DeviceMesh,
    *,
    parallel: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distribute the prompt and cache positions for the chosen strategy.

    DP shards the prompt along the batch dim; TP replicates it (same prompt on
    every chip). `cache_position` is replicated in both cases - it indexes the
    sequence dim, which is never sharded.
    """
    ids_placements = [Shard(0)] if parallel == "dp" else [Replicate()]
    dids = distribute_tensor(input_ids.to("tt"), mesh, ids_placements)
    dpos = distribute_tensor(cache_position.to("tt"), mesh, [Replicate()])
    return dids, dpos
