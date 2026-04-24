# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chisel bind/unbind — one-call setup and teardown for builder integration."""

from typing import Callable, Optional

from ttmlir.ir import Operation
from _ttmlir_runtime import runtime as tt_runtime

from .context import ChiselContext
from .callbacks import (
    chisel_pre_program_callback,
    chisel_post_program_callback,
    chisel_pre_op_callback,
    chisel_post_op_callback,
)


def bind(skip_criterion: Optional[Callable[[Operation], bool]] = None):
    """Initialize ChiselContext and register all 4 callbacks with DebugHooks.

    Args:
        skip_criterion: Optional predicate (Operation) -> bool. When it returns
            True for an op, chisel overwrites that op's device output tensor(s)
            with the isolation golden result after the device executes the op.
            When None (default), skip mode is disabled.
    """
    ctx = ChiselContext()
    ctx.skip_criterion = skip_criterion
    tt_runtime.DebugHooks.get(
        pre_op=chisel_pre_op_callback,
        post_op=chisel_post_op_callback,
        pre_program=chisel_pre_program_callback,
        post_program=chisel_post_program_callback,
    )


def unbind():
    """Tear down ChiselContext singleton. Safe to call even if bind() was not called."""
    tt_runtime.DebugHooks.get()
    if ChiselContext._instance is not None:
        ChiselContext._instance.close_results()
    ChiselContext.reset_instance()
