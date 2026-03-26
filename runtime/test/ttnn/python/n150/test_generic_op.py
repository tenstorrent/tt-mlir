# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from ttrt.common.util import *
from ..utils import (
    load_binary,
    DeviceContext,
    ProgramTestRunner,
    get_flatbuffer_base_path,
)

FLATBUFFER_BASE_PATH = get_flatbuffer_base_path("Silicon", "TTNN", "n150", "generic_op")


def test_generic_op_abs():
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "generic_op.mlir.tmp.ttnn")
    binary = load_binary(binary_path)

    test_runner = ProgramTestRunner(
        binary,
        compute_golden=lambda inputs: (abs(inputs[0])),
    )

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

        test_runner.run_program_and_compare_golden(
            device,
            inputs_runtime_with_layout,
            golden,
        )
