# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Run a flatbuffer binary on device via the builder runtime (execute_fb) with
chisel's DebugHooks callbacks registered. Each op's check is a named pytest
subtest so all failures are reported without aborting on the first mismatch.

Usage:
    pytest test/python/chisel/test_device_execution.py --binary path/to/model.ttnn -v
"""
import functools
import logging

from __ttmlir_runtime import runtime as tt_runtime
from builder.base.builder_runtime import execute_fb
from chisel import bind, unbind
from chisel.callbacks import chisel_post_op_callback
from chisel.context import ChiselContext

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _wrap_post_op_for_subtests(subtests):
    """Wrap chisel_post_op_callback so each op is reported as its own pytest
    subtest. The subtest fixture catches any AssertionError raised on mismatch,
    records the failure, and continues to the next op."""

    @functools.wraps(chisel_post_op_callback)
    def wrapper(binary, program_context, op_context):
        program = ChiselContext.get_instance().current_program
        op_name = (
            program.current_op.name if program and program.current_op else "unknown"
        )
        with subtests.test(op=op_name):
            chisel_post_op_callback(binary, program_context, op_context)

    return wrapper


def test_device_execution(subtests, binary_path, device):
    """Execute a flatbuffer on device via builder's execute_fb; mismatches are
    reported as per-op pytest subtests."""

    bind()  # reset ChiselContext, register all 4 chisel callbacks via DebugHooks
    # DebugHooks.get() merges per-slot, so this overwrites only post_op while
    # leaving the pre_op / pre_program / post_program hooks that bind() set.
    tt_runtime.DebugHooks.get(post_op=_wrap_post_op_for_subtests(subtests))

    try:
        execute_fb(
            compiled_bin=binary_path,
            device=device,
            disable_golden=True,  # chisel handles golden comparison via callbacks
            enable_intermediate_verification=False,  # prevent execute_fb re-registering post_op
        )
    except Exception as e:
        logger.error("Device execution failed for %s: %s", binary_path, e)
        raise
    finally:
        unbind()
