# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Run a flatbuffer binary on device with chisel in strict mode. Each op's shape
check is a named pytest subtest so all failures are reported without aborting
on the first mismatch.

Usage:
    pytest tools/chisel/tests/test_device_execution.py --binary path/to/model.ttnn -v
"""
from ttrt.common.api import API

from chisel.callbacks import chisel_post_op_callback, chisel_pre_op_callback, with_pytest_subtests
from chisel.context import ChiselContext


def test_device_execution(subtests, binary_path):
    from ttrt import runtime as tt_runtime
    """Execute a flatbuffer on device; shape mismatches are reported as subtests."""
    ChiselContext.reset_instance()
    ctx = ChiselContext.get_instance()
    ctx.strict = True
    tt_runtime.DebugHooks.get(
        chisel_pre_op_callback,
        with_pytest_subtests(subtests)(chisel_post_op_callback),
    )

    try:
        API.initialize_apis()
        run = API.Run(
            {
                "binary": binary_path,
                "--disable-ttrt-callbacks": True,
                "--program-index": "all",
            }
        )
        run()
    finally:
        ChiselContext.reset_instance()
