# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Skip predicates: decide, per op, whether to substitute its device output.

A skip predicate is a ``Callable[[ChiselContext], bool]`` consulted in post-op
*after* the PCC checks (so ``ctx.op``, ``ctx.output_refs`` and the op's numerics
verdict are all available). When it returns True and accumulation mode is on,
chisel overwrites the op's device output with the isolation golden
(``golden(device_inputs)``) - simulating the op executing correctly. See
``callbacks._skip_op``.

Set the active predicate via ``ChiselChecksConfig.skip_op``.
"""
from typing import Callable

from .ops import get_op_outputs
from .report import NumericsMode, RecordStatus

SkipPredicate = Callable[["ChiselContext"], bool]


def skip_op_names(*op_names: str) -> SkipPredicate:
    """Skip ops whose operation name (e.g. ``"ttnn.matmul"``) is in `op_names`."""
    wanted = frozenset(op_names)

    def predicate(ctx: "ChiselContext") -> bool:
        return ctx.op.name in wanted

    return predicate


def skip_ssa(*names: str) -> SkipPredicate:
    """Skip ops with an output SSA name (e.g. ``"%5"``) in `names`."""
    wanted = frozenset(names)

    def predicate(ctx: "ChiselContext") -> bool:
        asm_state = ctx.asm_state
        return any(out.get_name(asm_state) in wanted for out in get_op_outputs(ctx.op))

    return predicate


def skip_on_bad_pcc(mode: NumericsMode = NumericsMode.ACCUMULATED) -> SkipPredicate:
    """Skip ops whose numerics check under `mode` failed (PCC/atol/rtol).

    Reads the op's retained numerics payloads (`ctx.op_numerics`); a custom
    predicate can inspect the same payloads for finer control (e.g. a pcc
    threshold stricter than the configured one).
    """

    def predicate(ctx: "ChiselContext") -> bool:
        return any(
            p.mode == mode and p.status is RecordStatus.NUMERICS_FAIL
            for p in (ctx.op_numerics or [])
        )

    return predicate


def any_of(*preds: SkipPredicate) -> SkipPredicate:
    """Skip if any of `preds` returns True (OR)."""

    def predicate(ctx: "ChiselContext") -> bool:
        return any(p(ctx) for p in preds)

    return predicate


def all_of(*preds: SkipPredicate) -> SkipPredicate:
    """Skip only if every one of `preds` returns True (AND)."""

    def predicate(ctx: "ChiselContext") -> bool:
        return all(p(ctx) for p in preds)

    return predicate
