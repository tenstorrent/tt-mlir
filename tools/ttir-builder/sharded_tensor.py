# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import operator
from functools import reduce
from typing import Any, Iterable, Iterator, List, Tuple, Union, Dict

import torch

# ----------------------------------------------------------------------
# metadata-only attr whitelist
# ----------------------------------------------------------------------
SAFE_TENSOR_ATTRS = {
    "shape",
    "dtype",
    "device",
    "layout",
    "requires_grad",
    "is_cuda",
    "is_floating_point",
    "is_complex",
    "ndim",
    "dim",
    "size",
    "stride",
    "storage_offset",
    "numel",
    "element_size",
    "is_contiguous",
    "is_pinned",
}


# ----------------------------------------------------------------------
# helper: row-major ravel
# ----------------------------------------------------------------------
def _ravel_nd(idx: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    flat, mult = 0, 1
    for size, i in zip(reversed(shape), reversed(idx)):
        flat += i * mult
        mult *= size
    return flat


# ----------------------------------------------------------------------
# walk util (still used by __torch_function__)
# ----------------------------------------------------------------------
def _walk(*trees) -> Iterable:
    for tree in trees:
        if isinstance(tree, (list, tuple)):
            for v in tree:
                yield from _walk(v)
        elif isinstance(tree, dict):
            for v in tree.values():
                yield from _walk(v)
        else:
            yield tree


# ----------------------------------------------------------------------
# main class
# ----------------------------------------------------------------------
class ShardedTensor:
    def __init__(self, shards: List[torch.Tensor], shard_shape: Tuple[int, ...]):
        if not shards:
            raise ValueError("shards list must be non-empty")
        if reduce(operator.mul, shard_shape, 1) != len(shards):
            raise ValueError("len(shards) must equal product(shard_shape)")

        first = shards[0]
        for t in shards[1:]:
            if t.shape != first.shape:
                raise ValueError("all shards must share the same shape")
            if t.dtype != first.dtype:
                raise ValueError("all shards must share the same dtype")
            if t.device != first.device:
                raise ValueError("all shards must reside on the same device")

        self._shards: List[torch.Tensor] = shards
        self._shard_shape: Tuple[int, ...] = shard_shape

    # ------------------------------------------------------------
    # read-only attr forwarding
    # ------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in SAFE_TENSOR_ATTRS:
            return getattr(self._shards[0], name)
        raise AttributeError(
            f"'ShardedTensor' object has no attribute '{name}'. "
            "For mutating ops use torch.* functions."
        )

    # ------------------------------------------------------------
    # torch namespace hook
    # ------------------------------------------------------------
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not any(issubclass(t, ShardedTensor) for t in types):
            return func(*args, **kwargs)

        st_inputs = [a for a in _walk(args, kwargs) if isinstance(a, ShardedTensor)]
        n_shards = {len(st._shards) for st in st_inputs}
        if len(n_shards) != 1:
            raise RuntimeError("all ShardedTensors must have the same number of shards")
        n = n_shards.pop()

        def _take(arg, idx):
            if isinstance(arg, ShardedTensor):
                return arg._shards[idx]
            if isinstance(arg, (list, tuple)):
                return type(arg)(_take(a, idx) for a in arg)
            if isinstance(arg, dict):
                return {k: _take(v, idx) for k, v in arg.items()}
            return arg

        out_shards = [func(*_take(args, i), **_take(kwargs, i)) for i in range(n)]

        if all(isinstance(o, torch.Tensor) for o in out_shards):
            first_out = out_shards[0]
            if all(
                (o.shape, o.dtype, o.device)
                == (first_out.shape, first_out.dtype, first_out.device)
                for o in out_shards[1:]
            ):
                return ShardedTensor(out_shards, st_inputs[0]._shard_shape)
        return out_shards

    # ------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------
    @property
    def shards(self) -> List[torch.Tensor]:
        return self._shards

    @property
    def shard_shape(self) -> Tuple[int, ...]:
        return self._shard_shape

    def clone(self) -> "ShardedTensor":
        return ShardedTensor([t.clone() for t in self._shards], self._shard_shape)

    def contiguous(self) -> "ShardedTensor":
        return ShardedTensor([t.contiguous() for t in self._shards], self._shard_shape)

    # ------------------------------------------------------------
    # NEW: replica group clustering
    # ------------------------------------------------------------
    def replica_groups(self, cluster_axis: int) -> List[List[torch.Tensor]]:
        """
        Return shard clusters that vary `cluster_axis` while fixing others.
        Example: shape=(2,4), axis=0 -> [[t0,t4],[t1,t5],[t2,t6],[t3,t7]]
        """
        rank = len(self._shard_shape)
        if cluster_axis < 0:
            cluster_axis += rank
        if not 0 <= cluster_axis < rank:
            raise ValueError("cluster_axis out of range")

        other_axes = [ax for ax in range(rank) if ax != cluster_axis]
        groups: List[List[torch.Tensor]] = []
        for fixed in itertools.product(
            *[range(self._shard_shape[ax]) for ax in other_axes]
        ):
            group: Dict[int, torch.Tensor] = {}
            for v in range(self._shard_shape[cluster_axis]):
                coord = list(fixed)
                coord.insert(cluster_axis, v)
                idx = _ravel_nd(tuple(coord), self._shard_shape)
                group[idx] = self._shards[idx]
            groups.append(group)
        return groups

    # ------------------------------------------------------------
    # misc dunders
    # ------------------------------------------------------------
    def __len__(self) -> int:
        return self._shards[0].size(0)

    def __iter__(self) -> Iterator:
        return iter(self._shards[0])

    def __repr__(self) -> str:
        return (
            "ShardedTensor("
            f"num_shards={len(self._shards)}, "
            f"shard_shape={self._shard_shape}, "
            f"tensor_shape={tuple(self._shards[0].shape)}, "
            f"dtype={self._shards[0].dtype}, "
            f"device={self._shards[0].device})"
        )


# ----------------------------------------------------------------------
# alias used elsewhere
# ----------------------------------------------------------------------
TensorLike = Union[torch.Tensor, ShardedTensor]
