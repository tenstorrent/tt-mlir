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
    Storage,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
    get_torch_output_container,
    assert_pcc,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/consteval/Output"
)


def test_system_desc(helper: Helper, request):
    ttrt.runtime.set_mlir_home(TT_MLIR_HOME)
    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Distributed)
    ttrt.runtime.launch_distributed_runtime()
    system_desc = ttrt.runtime.get_current_system_desc()
    assert system_desc is not None
    ttrt.runtime.shutdown_distributed_runtime()

    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Local)
    system_desc_local = ttrt.runtime.get_current_system_desc()
    assert system_desc.as_json() == system_desc_local.as_json()


@pytest.mark.parametrize("num_loops", [64])
def test_flatbuffer_execution(helper: Helper, request, num_loops):
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "binary_ops.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops_distributed",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Binary ops distributed test",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    ttrt.runtime.set_mlir_home(TT_MLIR_HOME)
    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Distributed)
    ttrt.runtime.launch_distributed_runtime()

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden = test_runner.get_inputs_and_golden(
            device, borrow=False
        )
        for i in range(num_loops):
            test_runner.run_program_and_compare_golden(
                device, inputs_runtime_with_layout, golden
            )

        outputs = []
        for i in range(num_loops):
            output = test_runner.run_program(device, inputs_runtime_with_layout)
            outputs.append(output)

        for output in outputs:
            output_torch = get_torch_output_container(test_runner.program)
            ttrt.runtime.memcpy(output_torch.data_ptr(), output)
            assert_pcc(output_torch, golden)

    ttrt.runtime.shutdown_distributed_runtime()
    helper.teardown()
    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Local)
