# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chisel bind/unbind — one-call setup and teardown for builder integration."""

from .context import ChiselContext
from .callbacks import (
    chisel_pre_op_callback,
    chisel_post_op_callback,
)


def bind():
    """Initialize ChiselContext and register op callbacks with DebugHooks."""
    import _ttmlir_runtime as tt_runtime

    ChiselContext()
    tt_runtime.runtime.DebugHooks.get(
        chisel_pre_op_callback,
        chisel_post_op_callback,
    )


def unbind():
    """Tear down ChiselContext singleton. Safe to call even if bind() was not called."""
    ChiselContext.reset_instance()
