# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import ttrt
import ttrt.runtime
from ttrt.common.util import *
from ...utils import (
    TT_MLIR_HOME,
    TT_METAL_HOME_EXTERNAL,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
    subprocess_get_system_descriptor,
    get_torch_output_container,
    assert_pcc,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/llmbox/binary/Output"
)

RANK_BINDING_PATH = f"{TT_METAL_HOME_EXTERNAL}/tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"


@pytest.mark.xfail(
    reason="TODO(#5320): System descriptor returned is local per host process, we need unification logic to merge them"
)
def test_system_desc(request):
    system_desc_local = subprocess_get_system_descriptor(request)

    assert os.path.exists(
        RANK_BINDING_PATH
    ), f"Rank binding path not found: {RANK_BINDING_PATH}"

    ttrt.runtime.set_mlir_home(TT_MLIR_HOME)
    ttrt.runtime.set_metal_home(TT_METAL_HOME_EXTERNAL)

    mp_args = ttrt.runtime.MultiProcessArgs.create(RANK_BINDING_PATH)
    mp_args.with_allow_run_as_root(True)

    distributed_options = ttrt.runtime.DistributedOptions()
    distributed_options.mode = ttrt.runtime.DistributedMode.MultiProcess
    distributed_options.multi_process_args = mp_args

    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Distributed)
    ttrt.runtime.launch_distributed_runtime(distributed_options)
    system_desc = ttrt.runtime.get_current_system_desc()
    assert system_desc is not None
    ttrt.runtime.shutdown_distributed_runtime()

    assert system_desc.as_json() == system_desc_local.as_json()


@pytest.mark.parametrize("num_loops", [64])
@pytest.mark.parametrize("mesh_shape", ["1x8", "2x4"])
def test_flatbuffer_execution(request, num_loops, mesh_shape):
    assert os.path.exists(
        RANK_BINDING_PATH
    ), f"Rank binding path not found: {RANK_BINDING_PATH}"

    mesh_shape_list = list(map(int, mesh_shape.split("x")))
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, f"simple_add_{mesh_shape}.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"

    test_config = ProgramTestConfig(
        name="simple_add_distributed",
        expected_num_inputs=2,
        compute_golden=lambda inputs: (inputs[0] + inputs[1]),
        description="Simple add distributed test",
    )

    logger = Logger()
    file_manager = FileManager(logger)
    binary = Binary(logger, file_manager, binary_path)

    curr_system_desc = json.loads(subprocess_get_system_descriptor(request).as_json())
    binary_system_desc = binary.system_desc_dict

    assert (
        curr_system_desc["system_desc"] == binary_system_desc
    ), "System descriptor mismatch"

    test_runner = ProgramTestRunner(test_config, binary, 0)

    ttrt.runtime.set_mlir_home(TT_MLIR_HOME)
    ttrt.runtime.set_metal_home(TT_METAL_HOME_EXTERNAL)

    mp_args = ttrt.runtime.MultiProcessArgs.create(RANK_BINDING_PATH)
    mp_args.with_allow_run_as_root(True)

    distributed_options = ttrt.runtime.DistributedOptions()
    distributed_options.mode = ttrt.runtime.DistributedMode.MultiProcess
    distributed_options.multi_process_args = mp_args

    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Distributed)
    ttrt.runtime.launch_distributed_runtime(distributed_options)

    with DeviceContext(mesh_shape=mesh_shape_list) as device:
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
    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Local)
