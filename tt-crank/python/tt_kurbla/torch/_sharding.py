"""DTensor op sharding strategies the tt backend needs but torch lacks.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding


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
    """Register the strategies above on the DTensor sharding propagator.

    Both the in-place and functional `index_copy` overloads appear depending on
    whether the write is traced (compile) or run eagerly.
    """
    aten = torch.ops.aten
    register_sharding(aten.index_copy_.default)(_index_copy_sharding)
    register_sharding(aten.index_copy.default)(_index_copy_sharding)
