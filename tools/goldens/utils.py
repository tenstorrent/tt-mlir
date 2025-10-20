# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Golden function mappings for TTIR, TTNN, D2M, and StableHLO operations.

This module provides a centralized mapping between operations and their
corresponding PyTorch golden reference implementations. Each golden function
serves as a reference implementation that produces the expected output for
comparison with operation results.
"""

from __future__ import annotations
from typing import Dict, Callable, Any, Optional, Union, List, Tuple, Iterable, Iterator
import itertools
import operator
import torch
import torch.nn.functional
from ttmlir.dialects import ttir, stablehlo, d2m
from ttmlir.ir import (
    Attribute,
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    BoolAttr,
    DenseI32ArrayAttr,
    DenseI64ArrayAttr,
)


class GoldenFunction:
    """
    GoldenFunction is a utility class for managing golden values for single device or multi device tensors.
    GoldenFunction represents a list of torch.Tensor objects, each representing a shard of a tensor.
    For single device tensors, it contains a single shard.

    How is this class compatible with torch.* operations?
      * For read-only tensor attributes (like shape, dtype, device, etc.), GoldenFunction forwards attribute access to the first shard.
      * For torch.* operations, the class implements the `__torch_function__` protocol.
      * When a torch function is called on a GoldenFunction, the function is applied independently to each shard.
      * The results are collected into a new GoldenFunction.
    """

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
        "is_quantized",
    }

    # Mutating methods. We will always return a new GoldenFunction to avoid in-place mutations by design.
    _MUTATING_METHOD_NAMES = [
        "to",
        "transpose",
        "reshape",
        "repeat",
        "permute",
        "flatten",
        "squeeze",
        "unsqueeze",
        "float",
        "clamp",
        "int_repr",
        "detach",
        "requires_grad_",
        "long",
    ]
    _MUTATING_METHODS = {
        name: (
            lambda name: lambda shard, *args, **kwargs: getattr(shard, name)(
                *args, **kwargs
            )
        )(name)
        for name in _MUTATING_METHOD_NAMES
    }

    # ----- Methods -----

    def __init__(self, shard_map: Dict[int, torch.Tensor], mesh_shape: Tuple[int, int]):
        it = iter(shard_map.values())
        first = next(it)

        for t in it:
            if t.shape != first.shape:
                raise ValueError(f"Shape mismatch: {t.shape} != {first.shape}")
            if t.dtype != first.dtype:
                raise ValueError(f"Dtype mismatch: {t.dtype} != {first.dtype}")
            if t.device != first.device:
                raise ValueError(f"Device mismatch: {t.device} != {first.device}")

        self._shard_map = shard_map
        self._mesh_shape = mesh_shape

    # ----- Private static methods -----
    def __getitem__(self, key: int) -> GoldenFunction:
        out_shards = {k: v.__getitem__(key) for k, v in self._shard_map.items()}
        ref = next(iter(out_shards.values()))
        if not all(isinstance(t, torch.Tensor) for t in out_shards.values()):
            return out_shards
        # Wrap
        return GoldenFunction(out_shards, self.mesh_shape)

    def _binary_map(self, other, op):
        if isinstance(other, GoldenFunction):
            keys = sorted(self._shard_map.keys())
            if set(keys) != set(other._shard_map.keys()):
                raise RuntimeError("Shard key mismatch between operands.")
            out_shards = {k: op(self._shard_map[k], other._shard_map[k]) for k in keys}
        else:
            out_shards = {k: op(t, other) for k, t in self._shard_map.items()}
        # Always wrap (even 0-D)
        return GoldenFunction(out_shards, self._mesh_shape)

    def __lt__(self, other):
        return self._binary_map(other, operator.lt)

    def __le__(self, other):
        return self._binary_map(other, operator.le)

    def __gt__(self, other):
        return self._binary_map(other, operator.gt)

    def __ge__(self, other):
        return self._binary_map(other, operator.ge)

    def __eq__(self, other):
        return self._binary_map(other, operator.eq)

    def __ne__(self, other):
        return self._binary_map(other, operator.ne)

    def __add__(self, other):
        return self._binary_map(other, operator.add)

    def __radd__(self, other):
        return self._binary_map(other, operator.add)

    def __sub__(self, other):
        return self._binary_map(other, operator.sub)

    def __rsub__(self, other):
        return self._binary_map(other, lambda a, b: operator.sub(b, a))

    @staticmethod
    def _walk_tree(*trees) -> Iterable:
        # Yield leaves in a nested structure.
        for tree in trees:
            if isinstance(tree, (list, tuple)):
                for v in tree:
                    yield from GoldenFunction._walk_tree(v)
            elif isinstance(tree, dict):
                for v in tree.values():
                    yield from GoldenFunction._walk_tree(v)
            else:
                yield tree

    # ----- Private methods -----

    def __getattr__(self, name: str) -> Any:
        if name in GoldenFunction._SAFE_TENSOR_ATTRS:
            return getattr(self._shard_map[0], name)
        elif name in GoldenFunction._MUTATING_METHODS:

            def method(*args, **kwargs):
                func = GoldenFunction._MUTATING_METHODS[name]
                return self.apply_shardwise(lambda shard: func(shard, *args, **kwargs))

            return method
        raise AttributeError(
            f"'GoldenFunction' has no attribute '{name}'. "
            "For mutating ops call torch.* directly."
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not any(issubclass(t, cls) for t in types):
            return NotImplemented

        # Collect all GoldenFunction inputs.
        st_inputs = [
            a for a in GoldenFunction._walk_tree(args, kwargs) if isinstance(a, cls)
        ]
        shard_counts = {len(st.shard_map) for st in st_inputs}
        if len(shard_counts) != 1:
            raise RuntimeError("All GoldenFunctions must have the same shard count.")

        # All shard_maps should share the same set of keys.
        shard_keys = [set(st.shard_map.keys()) for st in st_inputs]
        if not all(keys == shard_keys[0] for keys in shard_keys[1:]):
            raise RuntimeError(
                "All GoldenFunctions must have the same shard keys (devices amongst which the GoldenFunction lives)."
            )

        keys = sorted(shard_keys[0])  # deterministic order

        def _take(obj, k: int):
            if isinstance(obj, cls):
                return obj.shard_map[k]
            if isinstance(obj, (list, tuple)):
                return type(obj)(_take(v, k) for v in obj)
            if isinstance(obj, dict):
                return {kk: _take(v, k) for kk, v in obj.items()}
            return obj

        # Apply func shard-wise.
        out_shards = {k: func(*_take(args, k), **_take(kwargs, k)) for k in keys}

        # Check if output is a tuple of tensors
        if isinstance(next(iter(out_shards.values())), tuple):
            # Wrap each element of the tuple separately
            tuple_len = len(next(iter(out_shards.values())))
            wrapped_tuple = []
            for i in range(tuple_len):
                element_shards = {k: out_shards[k][i] for k in keys}
                wrapped_tuple.append(cls(element_shards, st_inputs[0].mesh_shape))
            return tuple(wrapped_tuple)

        # If all results are Tensors and compatible, wrap back into GoldenFunction.
        if all(isinstance(o, torch.Tensor) for o in out_shards.values()):
            ref = next(iter(out_shards.values()))
            if all(
                (o.shape, o.dtype, o.device) == (ref.shape, ref.dtype, ref.device)
                for o in out_shards.values()
            ):
                return cls(out_shards, st_inputs[0].mesh_shape)

        return out_shards

    # ----- Public methods -----

    @property
    def shard_map(self) -> Dict[int, torch.Tensor]:
        return self._shard_map

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        return self._mesh_shape

    def zeros_like_builder(self, shape) -> GoldenFunction:
        shard_map = {}
        for device_id, shard in self.shard_map.items():
            shard_map[device_id] = torch.zeros(shape, dtype=shard.dtype)
        return GoldenFunction(shard_map, self.mesh_shape)

    def apply_shardwise(input_tensor: GoldenFunction, func: Callable) -> GoldenFunction:
        shard_map = {}
        for device_id, shard in input_tensor.shard_map.items():
            output_shard = shard.clone()
            shard_map[device_id] = func(output_shard)
        return GoldenFunction(shard_map, input_tensor.mesh_shape)

    def shard_at(self, device_id: int) -> GoldenFunction:
        if device_id not in self._shard_map:
            raise KeyError(f"Device ID {device_id} not found in shard map.")
        return self._shard_map[device_id]

    def clone(self) -> GoldenFunction:
        shard_map = {
            device_id: shard.clone() for device_id, shard in self.shard_map.items()
        }
        return GoldenFunction(shard_map, self.mesh_shape)

    def contiguous(self) -> GoldenFunction:
        return GoldenFunction(
            {k: t.contiguous() for k, t in self._shard_map.items()}, self.mesh_shape
        )

    def group_by_axis(self, axis: int) -> List[Dict[int, torch.Tensor]]:
        rows, cols = self._mesh_shape
        shard_map = self._shard_map

        grouped: List[Dict[int, torch.Tensor]] = []
        if axis == 1:
            for r in range(rows):
                row_group: Dict[int, torch.Tensor] = {}
                for c in range(cols):
                    idx = r * cols + c
                    row_group[idx] = shard_map[idx]
                grouped.append(row_group)
        else:
            for c in range(cols):
                col_group: Dict[int, torch.Tensor] = {}
                for r in range(rows):
                    idx = r * cols + c
                    col_group[idx] = shard_map[idx]
                grouped.append(col_group)

        return grouped


def unpack_mlir_attr(attr):
    """Unpack MLIR attributes into plain Python values.

    Supports IntegerAttr, BoolAttr, DenseI32ArrayAttr, DenseI64ArrayAttr, ArrayAttr,
    as well as native Python list/tuple/int/bool. Raises ValueError for unsupported types.
    """
    if isinstance(attr, IntegerAttr):
        return attr.value
    if isinstance(attr, BoolAttr):
        return attr.value
    if isinstance(attr, (DenseI64ArrayAttr, DenseI32ArrayAttr)):
        return list(attr)
    if isinstance(attr, ArrayAttr):
        return [unpack_mlir_attr(item) for item in attr]
    if isinstance(attr, (list, tuple)):
        return list(attr)
    if isinstance(attr, (int, bool)):
        return attr
    raise ValueError(f"Unexpected attribute type: {type(attr)}")


def get_golden_function(op_class: type, **kwargs) -> Optional[Callable]:
    """
    Get the golden function for a given operation class.

    Parameters
    ----------
    op_class : type
        The operation class (e.g., ttir.AbsOp)
    **kwargs
        Additional keyword arguments for specialized operation selection

    Returns
    -------
    Optional[Callable]
        The corresponding golden function, or None if not found
    """
    from .mappings import GOLDEN_MAPPINGS
    from .custom_goldens import tilize_golden, untilize_golden

    # Handle special cases with parameters
    if (
        op_class == ttir.ToLayoutOp or op_class == d2m.ToLayoutOp
    ) and "tilize" in kwargs:
        if kwargs["tilize"]:
            return tilize_golden
        else:
            return untilize_golden

    if op_class in GOLDEN_MAPPINGS:
        return GOLDEN_MAPPINGS[op_class]

    return None
