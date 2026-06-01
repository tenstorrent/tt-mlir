# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Infer index-range constraints for `func.func` arguments by walking backwards
from range-restricted consumer ops (embedding, gather, scatter, ...) through
value-preserving ops (reshape, permute, typecast, ...).

The single public entry point is `infer_arg_ranges(parsed_func)`, which returns
a `{arg_number: (low, high_exclusive)}` dict for any argument whose value is
provably constrained to a sub-range by a consumer. Arguments not in the dict
have no inferred constraint and should use the default random distribution.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Type

from ttmlir.ir import Block, OpView, Operation, RankedTensorType, Value
from ttmlir.dialects import func, ttir


Range = Tuple[int, int]  # [low, high_exclusive)


# -----------------------------------------------------------------------------
# Range-restricted consumer ops.
#
# Each entry maps an OpView class to a callable that yields (operand_value,
# range) pairs for the operands of that op that must lie in a restricted range.
# -----------------------------------------------------------------------------


def _embedding_constraints(op: OpView) -> Iterable[Tuple[Value, Range]]:
    # input indices must be in [0, weight.shape[0]).
    weight_shape = RankedTensorType(op.weight.type).shape
    if weight_shape and weight_shape[0] > 0:
        yield op.input, (0, weight_shape[0])


def _gather_or_scatter_constraints(op: OpView) -> Iterable[Tuple[Value, Range]]:
    # index must be in [0, input.shape[dim]).
    input_shape = RankedTensorType(op.input.type).shape
    dim = int(op.dim.value) if hasattr(op.dim, "value") else int(op.dim)
    if 0 <= dim < len(input_shape) and input_shape[dim] > 0:
        yield op.index, (0, input_shape[dim])


def _update_cache_constraints(op: OpView) -> Iterable[Tuple[Value, Range]]:
    # update_index value(s) must be in [0, cache.shape[2]) (the sequence dim).
    # Cache shape is [num_users, num_heads, max_seq_len, head_dim].
    cache_shape = RankedTensorType(op.cache.type).shape
    if len(cache_shape) >= 3 and cache_shape[2] > 0:
        yield op.update_index, (0, cache_shape[2])


_CONSUMER_TABLE: Dict[
    Type[OpView], Callable[[OpView], Iterable[Tuple[Value, Range]]]
] = {
    ttir.EmbeddingOp: _embedding_constraints,
    ttir.EmbeddingBackwardOp: _embedding_constraints,
    ttir.GatherOp: _gather_or_scatter_constraints,
    ttir.ScatterOp: _gather_or_scatter_constraints,
    ttir.UpdateCacheOp: _update_cache_constraints,
}


# -----------------------------------------------------------------------------
# Value-preserving ops: ops whose element values (and therefore their value
# range) are preserved up to a shape change. The walker follows `op.input`
# backwards through these.
# -----------------------------------------------------------------------------


_VALUE_PRESERVING_OPS: Set[Type[OpView]] = {
    ttir.ReshapeOp,
    ttir.PermuteOp,
    ttir.TransposeOp,
    ttir.BroadcastOp,
    ttir.SqueezeOp,
    ttir.UnsqueezeOp,
    ttir.SliceStaticOp,
    ttir.TypecastOp,
}


def _dest_can_hold_range(op: OpView, value_range: Range) -> bool:
    """True if the op's result dtype can represent every value in `value_range`.

    For non-typecast ops in the value-preserving set this is trivially true.
    For typecast we check the destination dtype's representable range against
    the inferred `[low, high)`.
    """
    if not isinstance(op, ttir.TypecastOp):
        return True

    result_type = RankedTensorType(op.result.type)
    elem = result_type.element_type
    low, high = value_range
    max_val = high - 1

    # MLIR integer-type interface: width + signedness probes.
    width = getattr(elem, "width", None)
    if width is None:
        # Floating point can easily represent small integer ranges; assume yes.
        return True

    is_signed = getattr(elem, "is_signed", False) or getattr(
        elem, "is_signless", False
    )
    if is_signed:
        type_min = -(1 << (width - 1))
        type_max = (1 << (width - 1)) - 1
    else:
        type_min = 0
        type_max = (1 << width) - 1

    return low >= type_min and max_val <= type_max


# -----------------------------------------------------------------------------
# Walker
# -----------------------------------------------------------------------------


def _intersect(a: Optional[Range], b: Range) -> Range:
    if a is None:
        return b
    return (max(a[0], b[0]), min(a[1], b[1]))


def _walk_back_to_arg(
    value: Value,
    value_range: Range,
    arg_index_by_value: Dict[Value, int],
    result: Dict[int, Range],
) -> None:
    """Walk backwards from `value` through value-preserving ops. If we reach
    a function argument, intersect `value_range` into `result[arg_idx]`.
    Anything else terminates the walk silently (no constraint propagated)."""
    visited: List[Value] = []
    current: Value = value
    current_range: Range = value_range

    while True:
        # Cycle guard (shouldn't happen in SSA, but cheap insurance).
        if any(current == v for v in visited):
            return
        visited.append(current)

        owner = current.owner
        if isinstance(owner, Block):
            arg_idx = arg_index_by_value.get(current)
            if arg_idx is not None:
                result[arg_idx] = _intersect(result.get(arg_idx), current_range)
            return

        # owner is an Operation; downcast to a typed OpView.
        op_view = owner.opview if isinstance(owner, Operation) else owner
        if type(op_view) not in _VALUE_PRESERVING_OPS:
            return
        if not _dest_can_hold_range(op_view, current_range):
            return

        # All ops in the value-preserving set expose their data operand as
        # `input` (verified for reshape/permute/transpose/broadcast/squeeze/
        # unsqueeze/slice_static/typecast).
        next_value = op_view.input
        if next_value is None:
            return
        current = next_value


def infer_arg_ranges(parsed_func: func.FuncOp) -> Dict[int, Range]:
    """Return {arg_number: (low, high_exclusive)} for each func argument that
    is constrained by a range-restricted consumer reachable through a chain of
    value-preserving ops. Arguments without an inferred constraint are absent
    from the result.

    Multi-block funcs and nested calls are out of scope; the walker only
    inspects the immediate top-level block(s) of `parsed_func`.
    """
    result: Dict[int, Range] = {}

    arg_index_by_value: Dict[Value, int] = {
        arg: idx for idx, arg in enumerate(parsed_func.arguments)
    }

    for block in parsed_func.body:
        for op in block.operations:
            handler = _CONSUMER_TABLE.get(type(op))
            if handler is None:
                continue
            for operand_value, value_range in handler(op):
                before = dict(result)
                _walk_back_to_arg(
                    operand_value, value_range, arg_index_by_value, result
                )
                newly = {
                    k: v for k, v in result.items() if before.get(k) != v
                }
                print(
                    f"[RANGE-DEBUG] consumer={type(op).__name__} "
                    f"requires range {value_range} on an operand; "
                    f"propagated to args -> {newly if newly else '(no arg reached)'}",
                    flush=True,
                )

    return result
