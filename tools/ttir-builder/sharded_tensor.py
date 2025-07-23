# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import operator
from functools import reduce
from typing import Any, Iterator, List, Tuple, Union
import torch

# ----------------------------------------------------------------------
# safe read-only tensor members
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
# main class
# ----------------------------------------------------------------------
class ShardedTensor:
    """
    Tiny tensor-like wrapper holding uniform shards.
    """

    # --------------------------------------------------------------
    # validation util kept inside the class namespace
    # --------------------------------------------------------------
    @staticmethod
    def validate_shards(shards: List[torch.Tensor]) -> None:
        """
        Ensure all shards share shape, dtype, and device.
        Raise ValueError if not.
        """
        if not shards:
            raise ValueError("shards list must be non-empty")
        ref = shards[0]
        for t in shards[1:]:
            if t.shape != ref.shape:
                raise ValueError("all shards must share the same shape")
            if t.dtype != ref.dtype:
                raise ValueError("all shards must share the same dtype")
            if t.device != ref.device:
                raise ValueError("all shards must reside on the same device")

    # --------------------------------------------------------------
    # ctor
    # --------------------------------------------------------------
    def __init__(self, shards: List[torch.Tensor], shard_shape: Tuple[int, ...]):
        if reduce(operator.mul, shard_shape, 1) != len(shards):
            raise ValueError("len(shards) must equal product(shard_shape)")
        ShardedTensor.validate_shards(shards)  # reuse static util

        self._shards: List[torch.Tensor] = shards
        self._shard_shape: Tuple[int, ...] = shard_shape

    # --------------------------------------------------------------
    # attribute forwarding (read-only)
    # --------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in SAFE_TENSOR_ATTRS:
            return getattr(self._shards[0], name)
        raise AttributeError(
            f"'ShardedTensor' object has no attribute '{name}'. "
            "For mutating ops use torch.* functions."
        )

    # --------------------------------------------------------------
    # torch function override (unchanged)
    # --------------------------------------------------------------
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # ... same as 이전 버전 ...
        if kwargs is None:
            kwargs = {}
        if not any(issubclass(t, ShardedTensor) for t in types):
            return func(*args, **kwargs)

        # collect inputs, validate shard counts
        st_inputs = [a for a in _walk(args, kwargs) if isinstance(a, ShardedTensor)]
        n_shards = {len(st._shards) for st in st_inputs}
        if len(n_shards) != 1:
            raise RuntimeError("all ShardedTensors must have the same number of shards")
        n = n_shards.pop()

        def _take(arg, idx: int):
            if isinstance(arg, ShardedTensor):
                return arg._shards[idx]
            if isinstance(arg, (list, tuple)):
                return type(arg)(_take(a, idx) for a in arg)
            if isinstance(arg, dict):
                return {k: _take(v, idx) for k, v in arg.items()}
            return arg

        out_shards = [func(*_take(args, i), **_take(kwargs, i)) for i in range(n)]

        if all(isinstance(o, torch.Tensor) for o in out_shards):
            ref = out_shards[0]
            if all(
                (o.shape, o.dtype, o.device) == (ref.shape, ref.dtype, ref.device)
                for o in out_shards[1:]
            ):
                return ShardedTensor(out_shards, st_inputs[0]._shard_shape)
        return out_shards

    # --------------------------------------------------------------
    # helpers, contiguous, clone, dunders (unchanged)
    # --------------------------------------------------------------
    @property
    def shards(self) -> List[torch.Tensor]:
        return self._shards

    @property
    def shard_shape(self) -> Tuple[int, ...]:
        return self._shard_shape

    def get_shard(self, idx: int) -> torch.Tensor:
        return self._shards[idx]

    def clone(self) -> "ShardedTensor":
        return ShardedTensor([t.clone() for t in self._shards], self._shard_shape)

    def contiguous(self) -> "ShardedTensor":
        return ShardedTensor([t.contiguous() for t in self._shards], self._shard_shape)

    def __len__(self) -> int:
        return self._shards[0].size(0)

    def __iter__(self) -> Iterator:
        return iter(self._shards[0])

    def __repr__(self) -> str:
        ref = self._shards[0]
        return (
            "ShardedTensor("
            f"num_shards={len(self._shards)}, "
            f"shard_shape={self._shard_shape}, "
            f"tensor_shape={tuple(ref.shape)}, "
            f"dtype={ref.dtype}, "
            f"device={ref.device})"
        )


# ----------------------------------------------------------------------
# internal walk util (unchanged)
# ----------------------------------------------------------------------
def _walk(*trees):
    for tree in trees:
        if isinstance(tree, (list, tuple)):
            for v in tree:
                yield from _walk(v)
        elif isinstance(tree, dict):
            for v in tree.values():
                yield from _walk(v)
        else:
            yield tree


TensorLike = Union[torch.Tensor, ShardedTensor]
