# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from ttrt.common.util import *
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Silicon/TTNN/n150/generic_op/Output"
)


def test_generic_op_abs(helper: Helper, request):
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "generic_op.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="generic_op_abs",
        expected_num_inputs=1,
        compute_golden=lambda inputs: (abs(inputs[0])),
        description="Generic op abs test",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden = test_runner.get_inputs_and_golden(device)

        test_runner.run_program_and_compare_golden(
            device,
            inputs_runtime_with_layout,
            golden,
        )

    helper.teardown()
