# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chisel bind/unbind — one-call setup and teardown for builder integration."""

import ttrt.runtime as tt_runtime
from chisel.context import ChiselContext
from chisel.callbacks import (
    chisel_pre_op_callback,
    chisel_post_op_callback,
)


def bind():
    """Initialize ChiselContext and register op callbacks with DebugHooks."""
    ChiselContext()
    tt_runtime.runtime.DebugHooks.get(
        chisel_pre_op_callback,
        chisel_post_op_callback,
    )


def unbind():
    """Tear down ChiselContext singleton. Safe to call even if bind() was not called."""
    ChiselContext.reset_instance()
