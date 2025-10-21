# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/consteval/Output"
)


@pytest.mark.parametrize("num_loops", [5])
def test_consteval_add_mul_subtract(helper: Helper, request, num_loops):
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "binary_ops.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops_consteval",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Binary ops consteval test",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden = test_runner.get_inputs_and_golden(device)
        for i in range(num_loops):
            # First execute should be a consteval cache miss
            # Subsequent executes should be consteval cache hit
            test_runner.run_program_and_compare_golden(
                device, inputs_runtime_with_layout, golden
            )
            assert debug_stats.get_stat("ConstEvalCacheMiss") == 1
            assert debug_stats.get_stat("ConstEvalCacheHit") == i

        ttrt.runtime.DebugStats.get().clear()

        inputs_runtime_with_layout, golden = test_runner.get_inputs_and_golden(device)

        for i in range(num_loops):
            # First execute should be a consteval cache miss because we've updated the inputs
            # Subsequent executes should be consteval cache hits
            test_runner.run_program_and_compare_golden(
                device,
                inputs_runtime_with_layout,
                golden,
            )
            assert debug_stats.get_stat("ConstEvalCacheMiss") == 1
            assert debug_stats.get_stat("ConstEvalCacheHit") == i

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()
