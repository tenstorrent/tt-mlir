# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator
from functools import reduce
from typing import Any, Iterable, Iterator, List, Tuple, Union

import torch

SAFE_TENSOR_ATTRS = {
    # pure metadata properties
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
    # memory layout / capacity (identical by invariant)
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
    A very small tensor-like wrapper holding uniform shards.

    Parameters
    ----------
    shards : list of torch.Tensor
        Flat list in row-major order.
    shard_shape : tuple of int
        Logical grid (e.g. (2, 2) for 4 shards).
    """

    # ------------------------------------------------------------------
    # ctor
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # attribute forwarding (read-only)
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:  # noqa: D401
        """
        Forward only safe read-only tensor attributes.

        Mutating tensor methods are *not* exposed on purpose.
        """
        if name in SAFE_TENSOR_ATTRS:
            return getattr(self._shards[0], name)
        raise AttributeError(
            f"'ShardedTensor' object has no attribute '{name}'. "
            "For mutating ops use torch.* functions."
        )

    # ------------------------------------------------------------------
    # torch function override
    # ------------------------------------------------------------------
    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: N802
        """
        Apply `func` shard-wise and wrap outputs when possible.
        """
        if kwargs is None:
            kwargs = {}

        # detect if call actually involves a ShardedTensor
        if not any(issubclass(t, ShardedTensor) for t in types):
            return func(*args, **kwargs)

        # all ShardedTensor inputs must agree on len(shards)
        st_inputs = [a for a in _walk(args, kwargs) if isinstance(a, ShardedTensor)]
        n_shards = {len(st._shards) for st in st_inputs}
        if len(n_shards) != 1:
            raise RuntimeError("all ShardedTensors must have the same number of shards")
        n = n_shards.pop()

        # helper: pick the i-th physical tensor for each arg
        def _take(arg, idx: int):
            if isinstance(arg, ShardedTensor):
                return arg._shards[idx]
            if isinstance(arg, (list, tuple)):
                return type(arg)(_take(a, idx) for a in arg)
            if isinstance(arg, dict):
                return {k: _take(v, idx) for k, v in arg.items()}
            return arg

        out_shards: List[torch.Tensor] = []
        for i in range(n):
            shard_args = _take(args, i)
            shard_kwargs = _take(kwargs, i)
            out_shards.append(func(*shard_args, **shard_kwargs))

        # attempt to wrap outputs back
        if all(isinstance(o, torch.Tensor) for o in out_shards):
            first_out = out_shards[0]
            if all(
                (o.shape, o.dtype, o.device)
                == (first_out.shape, first_out.dtype, first_out.device)
                for o in out_shards[1:]
            ):
                # preserve same shard layout as first input ShardedTensor
                shard_shape = st_inputs[0]._shard_shape
                return ShardedTensor(out_shards, shard_shape)

        # fallback: return list of outputs
        return out_shards

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    @property
    def shards(self) -> List[torch.Tensor]:
        "Return the underlying shard list."
        return self._shards

    @property
    def shard_shape(self) -> Tuple[int, ...]:
        "Return the logical shard grid shape."
        return self._shard_shape

    def get_shard(self, coord: Tuple[int, ...]) -> torch.Tensor:
        """
        Return shard at `coord` (row-major).
        """
        if len(coord) != len(self._shard_shape):
            raise ValueError("coord rank mismatch")
        idx = 0
        stride = 1
        for c, extent in zip(reversed(coord), reversed(self._shard_shape)):
            if not 0 <= c < extent:
                raise IndexError("coord out of bounds")
            idx += c * stride
            stride *= extent
        return self._shards[idx]

    def clone(self) -> ShardedTensor:
        return ShardedTensor([t.clone() for t in self._shards], self._shard_shape)

    # ------------------------------------------------------------------
    # contiguous
    # ------------------------------------------------------------------
    def contiguous(self) -> "ShardedTensor":
        """
        Return a new ShardedTensor whose shards are contiguous.
        """
        cont = [t.contiguous() for t in self._shards]
        return ShardedTensor(cont, self._shard_shape)

    # ------------------------------------------------------------------
    # dunder misc
    # ------------------------------------------------------------------
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
# internal utility
# ----------------------------------------------------------------------
def _walk(*trees):
    """
    Yield all leaves from nested tuples/lists/dicts.
    """
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
