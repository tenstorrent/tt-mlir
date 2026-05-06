# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chisel bind/unbind — one-call setup and teardown for builder integration."""

from _ttmlir_runtime import runtime as tt_runtime

from .context import ChiselContext, _UNSET
from .report import ChiselReport
from .callbacks import (
    chisel_pre_program_callback,
    chisel_post_program_callback,
    chisel_pre_op_callback,
    chisel_post_op_callback,
)


def bind():
    """Initialize ChiselContext and register all 4 callbacks with DebugHooks."""
    ChiselContext()
    tt_runtime.DebugHooks.get(
        pre_op=chisel_pre_op_callback,
        post_op=chisel_post_op_callback,
        pre_program=chisel_pre_program_callback,
        post_program=chisel_post_program_callback,
    )


def configure(
    *,
    strict=_UNSET,
    isolation_check=_UNSET,
    results_path=_UNSET,
    report_capacity=_UNSET,
):
    """Update flags on the live ChiselContext. Requires bind() first."""
    ChiselContext.get_instance().configure(
        strict=strict,
        isolation_check=isolation_check,
        results_path=results_path,
        report_capacity=report_capacity,
    )


def get_report() -> ChiselReport:
    """Return the live in-memory record buffer. Requires bind() first."""
    return ChiselContext.get_instance().report


def unbind():
    """Tear down ChiselContext singleton. Safe to call even if bind() was not called.

    Callers that need post-run access to the in-memory report should grab it
    via `get_report()` before calling unbind().
    """
    if ChiselContext._instance is not None:
        ChiselContext._instance.close_results()
    ChiselContext.reset_instance()

    tt_runtime.unregister_hooks()
