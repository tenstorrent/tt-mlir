# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chisel bind/unbind — one-call setup and teardown for builder integration."""
import _ttmlir_runtime as tt_runtime

from .context import ChiselContext
from .callbacks import (
    chisel_pre_program_callback,
    chisel_post_program_callback,
    chisel_pre_op_callback,
    chisel_post_op_callback,
)


def bind():
    """Initialize ChiselContext and register all 4 callbacks with DebugHooks."""
    ChiselContext()
    tt_runtime.runtime.DebugHooks.get(
        pre_op=chisel_pre_op_callback,
        post_op=chisel_post_op_callback,
        pre_program=chisel_pre_program_callback,
        post_program=chisel_post_program_callback,
    )


def unbind():
    """Tear down ChiselContext singleton. Safe to call even if bind() was not called."""
    tt_runtime.runtime.DebugHooks.get()
    ChiselContext.reset_instance()
