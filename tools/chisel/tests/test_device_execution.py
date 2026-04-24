# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Run a flatbuffer binary on device with chisel in strict mode. Each op's check
is a named pytest subtest so all failures are reported without aborting on the
first mismatch.

Usage:
    pytest tools/chisel/tests/test_device_execution.py --binary path/to/model.ttnn -v
"""
import functools
import logging

from _ttmlir_runtime import runtime as tt_runtime

from ttrt.common.api import API
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts

from chisel.callbacks import (
    chisel_pre_program_callback,
    chisel_post_program_callback,
    chisel_pre_op_callback,
    chisel_post_op_callback,
)
from chisel.context import ChiselContext

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = "/localdev/ndrakulic/tt-mlir/ttrt-artifacts"


def wrap_for_subtests(fn, subtests):
    """Wrap a post_op callback so each op runs as a named pytest subtest.

    Pairs with ctx.strict=True: the subtest fixture catches the AssertionError
    raised on mismatch, records the failure, and continues to the next op.
    """

    @functools.wraps(fn)
    def wrapper(binary, program_context, op_context):
        program = ChiselContext.get_instance().current_program
        op_name = program.current_op.name if program and program.current_op else "unknown"
        with subtests.test(op=op_name):
            fn(binary, program_context, op_context)

    return wrapper


def test_device_execution(subtests, binary_path):
    """Execute a flatbuffer on device; mismatches are reported as subtests."""
    ChiselContext.reset_instance()
    ctx = ChiselContext()
    ctx.strict = False
    # Keep a reference so the GC doesn't collect the callbacks before execution
    hooks = tt_runtime.DebugHooks.get(
        pre_op=chisel_pre_op_callback,
        post_op=wrap_for_subtests(chisel_post_op_callback, subtests),
        pre_program=chisel_pre_program_callback,
        post_program=chisel_post_program_callback,
    )
    assert hooks is not None, "Failed to register DebugHooks"

    try:
        rt_logger = RtLogger()
        rt_artifacts = RtArtifacts(
            logger=rt_logger, artifacts_folder_path=_ARTIFACTS_DIR
        )
        API.initialize_apis()
        run = API.Run(
            {
                "binary": binary_path,
                "--disable-ttrt-callbacks": True,
                "--log-file": "ttrt.log",
                "--print-input-output-tensors": True,
                "--save-artifacts": True,
            },
            logger=rt_logger,
            artifacts=rt_artifacts,
        )
        exit_code, result = run()
        assert exit_code == 0, result
    except Exception as e:
        logger.error("Device execution failed for %s: %s", binary_path, e)
        raise
    finally:
        ChiselContext.reset_instance()
        tt_runtime.unregister_hooks()
