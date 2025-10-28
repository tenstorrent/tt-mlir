# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import torch
import ttrt
import ttrt.runtime
from ttrt.common.util import *
from ...utils import (
    TT_MLIR_HOME,
    TT_METAL_RUNTIME_ROOT_EXTERNAL,
    Storage,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
    subprocess_get_system_descriptor,
    get_runtime_tensor_from_torch,
    get_torch_output_container,
    assert_pcc,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/llmbox/binary/Output"
)

RANK_BINDING_PATH = f"{TT_METAL_RUNTIME_ROOT_EXTERNAL}/tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"


def launch_distributed_runtime():
    assert os.path.exists(
        RANK_BINDING_PATH
    ), f"Rank binding path not found: {RANK_BINDING_PATH}"

    ttrt.runtime.set_mlir_home(TT_MLIR_HOME)
    ttrt.runtime.set_metal_home(TT_METAL_RUNTIME_ROOT_EXTERNAL)

    mp_args = ttrt.runtime.MultiProcessArgs.create(RANK_BINDING_PATH)
    mp_args.with_allow_run_as_root(True)

    distributed_options = ttrt.runtime.DistributedOptions()
    distributed_options.mode = ttrt.runtime.DistributedMode.MultiProcess
    distributed_options.multi_process_args = mp_args
    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Distributed)
    ttrt.runtime.launch_distributed_runtime(distributed_options)


def shutdown_distributed_runtime():
    ttrt.runtime.shutdown_distributed_runtime()
    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Local)


@pytest.mark.xfail(
    reason="TODO(#5320): System descriptor returned is local per host process, we need unification logic to merge them"
)
def test_system_desc(request):
    system_desc_local = subprocess_get_system_descriptor(request)

    launch_distributed_runtime()

    system_desc = ttrt.runtime.get_current_system_desc()
    assert system_desc is not None

    shutdown_distributed_runtime()

    assert system_desc.as_json() == system_desc_local.as_json()


def test_get_num_devices():
    launch_distributed_runtime()

    num_devices = ttrt.runtime.get_num_available_devices()
    assert num_devices == 8

    shutdown_distributed_runtime()


@pytest.mark.parametrize("mesh_shape", ["1x8"])
def test_get_mesh_shape(mesh_shape):
    launch_distributed_runtime()

    mesh_shape_list = list(map(int, mesh_shape.split("x")))

    with DeviceContext(mesh_shape=mesh_shape_list) as device:
        device_mesh_shape = device.get_mesh_shape()
        assert device_mesh_shape == mesh_shape_list

    shutdown_distributed_runtime()


def test_tensor_retain():
    launch_distributed_runtime()

    tensor = get_runtime_tensor_from_torch(torch.randn(1, 1), storage=Storage.Owned)
    tensor.set_retain(True)

    assert tensor.get_retain() == True

    tensor.set_retain(False)
    assert tensor.get_retain() == False

    shutdown_distributed_runtime()


def test_get_tensor_volume():
    launch_distributed_runtime()

    tensor = get_runtime_tensor_from_torch(torch.randn(177, 211), storage=Storage.Owned)
    assert tensor.get_volume() == 177 * 211

    shutdown_distributed_runtime()


def test_memcpy():
    launch_distributed_runtime()

    ones_tensor = torch.ones(177, 211)
    zeros_tensor = torch.zeros(177, 211)

    tensor1 = get_runtime_tensor_from_torch(ones_tensor, storage=Storage.Owned)
    tensor2 = get_runtime_tensor_from_torch(zeros_tensor, storage=Storage.Owned)

    # Copy from tensor1 to tensor2
    ttrt.runtime.memcpy(tensor2, tensor1)

    output_torch_tensor = torch.randn(177, 211)

    # Copy from tensor2 to output_torch_tensor
    ttrt.runtime.memcpy(output_torch_tensor.data_ptr(), tensor2)

    assert torch.allclose(output_torch_tensor, ones_tensor)

    shutdown_distributed_runtime()


def test_deallocate():
    launch_distributed_runtime()

    tensor = get_runtime_tensor_from_torch(torch.randn(177, 211), storage=Storage.Owned)

    ttrt.runtime.deallocate_tensor(tensor)
    assert not tensor.is_allocated()

    tensor = get_runtime_tensor_from_torch(torch.randn(177, 211), storage=Storage.Owned)
    tensor.set_retain(True)
    ttrt.runtime.deallocate_tensor(tensor)
    assert tensor.is_allocated()

    shutdown_distributed_runtime()


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

    launch_distributed_runtime()

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
            output = test_runner.submit_program(device, inputs_runtime_with_layout)
            outputs.append(output)

        for output in outputs:
            output_torch = get_torch_output_container(test_runner.program)
            output_host = ttrt.runtime.to_host(output, untilize=True, blocking=True)[0]
            ttrt.runtime.memcpy(output_torch.data_ptr(), output_host)
            assert_pcc(output_torch, golden)

    shutdown_distributed_runtime()


@pytest.mark.parametrize("num_loops", [64])
def test_flatbuffer_execution_dp(request, num_loops):
    assert os.path.exists(
        RANK_BINDING_PATH
    ), f"Rank binding path not found: {RANK_BINDING_PATH}"

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "simple_add_1x2.mlir.tmp.ttnn")

    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"

    test_config = ProgramTestConfig(
        name="simple_add_dp",
        expected_num_inputs=2,
        compute_golden=lambda inputs: (inputs[0] + inputs[1]),
        description="Simple add distributed data parallel test",
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

    launch_distributed_runtime()

    with DeviceContext(mesh_shape=[2, 4]) as device:

        submesh1 = ttrt.runtime.create_sub_mesh_device(
            device, mesh_shape=[1, 2], mesh_offset=[0, 1]
        )
        submesh2 = ttrt.runtime.create_sub_mesh_device(
            device, mesh_shape=[1, 2], mesh_offset=[1, 1]
        )

        (
            inputs_runtime_with_layout_submesh1,
            golden1,
        ) = test_runner.get_inputs_and_golden(submesh1, borrow=False)
        (
            inputs_runtime_with_layout_submesh2,
            golden2,
        ) = test_runner.get_inputs_and_golden(submesh2, borrow=False)

        # Synchronous back to back execution
        for i in range(num_loops):
            test_runner.run_program_and_compare_golden(
                submesh1, inputs_runtime_with_layout_submesh1, golden1
            )
            test_runner.run_program_and_compare_golden(
                submesh2, inputs_runtime_with_layout_submesh2, golden2
            )

        # Asynchronous data parallel execution
        outputs_submesh1 = []
        outputs_submesh2 = []
        for i in range(num_loops):
            output1 = test_runner.submit_program(
                submesh1, inputs_runtime_with_layout_submesh1
            )
            output2 = test_runner.submit_program(
                submesh2, inputs_runtime_with_layout_submesh2
            )
            outputs_submesh1.append(output1)
            outputs_submesh2.append(output2)

        for output in outputs_submesh1:
            output_torch = get_torch_output_container(test_runner.program)
            output_host = ttrt.runtime.to_host(output, untilize=True, blocking=True)[0]
            ttrt.runtime.memcpy(output_torch.data_ptr(), output_host)
            assert_pcc(output_torch, golden1)

        for output in outputs_submesh2:
            output_torch = get_torch_output_container(test_runner.program)
            output_host = ttrt.runtime.to_host(output, untilize=True, blocking=True)[0]
            ttrt.runtime.memcpy(output_torch.data_ptr(), output_host)
            assert_pcc(output_torch, golden2)

        ttrt.runtime.release_sub_mesh_device(submesh1)
        ttrt.runtime.release_sub_mesh_device(submesh2)

    shutdown_distributed_runtime()
