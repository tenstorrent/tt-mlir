# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import operator
from functools import reduce
from typing import Any, Iterable, Iterator, List, Tuple, Union, Dict

import torch

# Names that are safe to forward to the first shard.
_SAFE_TENSOR_ATTRS = {
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


class ShardedTensor:
    """A logical tensor split into equal-shaped shards."""

    # ------------------------------------------------------------
    # internal helpers (static)
    # ------------------------------------------------------------
    @staticmethod
    def _ravel_index(idx: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
        """Row-major flatten of an n-D index."""
        flat, stride = 0, 1
        for size, i in zip(reversed(shape), reversed(idx)):
            flat += i * stride
            stride *= size
        return flat

    @staticmethod
    def _walk_tree(*trees) -> Iterable:
        """Yield leaves in a nested structure."""
        for tree in trees:
            if isinstance(tree, (list, tuple)):
                for v in tree:
                    yield from ShardedTensor._walk_tree(v)
            elif isinstance(tree, dict):
                for v in tree.values():
                    yield from ShardedTensor._walk_tree(v)
            else:
                yield tree

    # ------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------
    def __init__(self, shards: List[torch.Tensor], shard_shape: Tuple[int, ...]):
        if not shards:
            raise ValueError("shards must be non-empty")
        if reduce(operator.mul, shard_shape, 1) != len(shards):
            raise ValueError("product(shard_shape) must equal len(shards)")

        first = shards[0]
        for t in shards[1:]:
            if t.shape != first.shape:
                raise ValueError("all shards must share the same shape")
            if t.dtype != first.dtype:
                raise ValueError("all shards must share the same dtype")
            if t.device != first.device:
                raise ValueError("all shards must share the same device")

        self._shards: List[torch.Tensor] = shards
        self._shard_shape: Tuple[int, ...] = shard_shape

    # ------------------------------------------------------------
    # attribute forwarding
    # ------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in _SAFE_TENSOR_ATTRS:
            return getattr(self._shards[0], name)
        raise AttributeError(
            f"'ShardedTensor' has no attribute '{name}'. "
            "For mutating ops call torch.* directly."
        )

    # ------------------------------------------------------------
    # torch dispatch
    # ------------------------------------------------------------
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not any(issubclass(t, cls) for t in types):
            return NotImplemented

        st_inputs = [a for a in cls._walk_tree(args, kwargs) if isinstance(a, cls)]
        shard_counts = {len(st._shards) for st in st_inputs}
        if len(shard_counts) != 1:
            raise RuntimeError("all ShardedTensors must have the same shard count")
        n = shard_counts.pop()

        def _take(obj, i: int):
            if isinstance(obj, cls):
                return obj._shards[i]
            if isinstance(obj, (list, tuple)):
                return type(obj)(_take(v, i) for v in obj)
            if isinstance(obj, dict):
                return {k: _take(v, i) for k, v in obj.items()}
            return obj

        out_shards = [func(*_take(args, i), **_take(kwargs, i)) for i in range(n)]

        if all(isinstance(o, torch.Tensor) for o in out_shards):
            ref = out_shards[0]
            if all(
                (o.shape, o.dtype, o.device) == (ref.shape, ref.dtype, ref.device)
                for o in out_shards[1:]
            ):
                return cls(out_shards, st_inputs[0]._shard_shape)

        return out_shards

    # ------------------------------------------------------------
    # public api
    # ------------------------------------------------------------
    @property
    def shards(self) -> List[torch.Tensor]:
        return self._shards

    def shard_at(self, idx: int) -> torch.Tensor:
        return self._shards[idx]

    @property
    def shard_shape(self) -> Tuple[int, ...]:
        return self._shard_shape

    def clone(self) -> "ShardedTensor":
        return ShardedTensor([t.clone() for t in self._shards], self._shard_shape)

    def contiguous(self) -> "ShardedTensor":
        return ShardedTensor([t.contiguous() for t in self._shards], self._shard_shape)

    def grouped_shards(self, axis: int) -> List[Dict[int, torch.Tensor]]:
        """Group shards that vary along *axis* while other coordinates are fixed."""
        rank = len(self._shard_shape)
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            raise ValueError("axis out of range")

        other_axes = [a for a in range(rank) if a != axis]
        groups: List[Dict[int, torch.Tensor]] = []
        for fixed in itertools.product(
            *[range(self._shard_shape[a]) for a in other_axes]
        ):
            group: Dict[int, torch.Tensor] = {}
            for v in range(self._shard_shape[axis]):
                coord = list(fixed)
                coord.insert(axis, v)
                idx = self._ravel_index(tuple(coord), self._shard_shape)
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
            f"{len(self._shards)} shards, "
            f"shard_shape={self._shard_shape}, "
            f"tensor_shape={tuple(self._shards[0].shape)}, "
            f"dtype={self._shards[0].dtype}, "
            f"device={self._shards[0].device})"
        )


TensorLike = Union[torch.Tensor, ShardedTensor]
