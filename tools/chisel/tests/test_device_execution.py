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
import logging

from ttrt.common.api import API
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts

from chisel.callbacks import (
    chisel_pre_program_callback,
    chisel_post_program_callback,
    chisel_pre_op_callback,
    chisel_post_op_callback,
    with_pytest_subtests,
)
from chisel.context import ChiselContext

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = "/localdev/ndrakulic/tt-mlir/ttrt-artifacts"


def test_device_execution(subtests, binary_path):
    """Execute a flatbuffer on device; mismatches are reported as subtests."""
    from ttrt import runtime as tt_runtime

    ChiselContext.reset_instance()
    ctx = ChiselContext()
    ctx.strict = False
    # Keep a reference so the GC doesn't collect the callbacks before execution
    hooks = tt_runtime.DebugHooks.get(
        pre_op=chisel_pre_op_callback,
        post_op=with_pytest_subtests(subtests)(chisel_post_op_callback),
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
