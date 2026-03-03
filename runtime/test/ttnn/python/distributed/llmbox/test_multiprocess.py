# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import torch
import ttrt
import ttrt.runtime
from typing import Dict, Any, Tuple, List
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


def compare_system_descriptors(
    desc1: Dict[str, Any], desc2: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    def _is_empty_container(value: Any) -> bool:
        return (isinstance(value, list) and len(value) == 0) or (
            isinstance(value, dict) and len(value) == 0
        )

    def _deep_compare(obj1: Any, obj2: Any, path: str, differences: List[str]) -> None:
        if type(obj1) != type(obj2):
            differences.append(
                f"{path}: Type mismatch - {type(obj1).__name__} vs {type(obj2).__name__}"
            )
            return

        # Multihost descriptors can have different chip channel assignments
        if path == "system_desc.chip_channels":
            return

        if isinstance(obj1, dict):
            keys1 = set(obj1.keys())
            keys2 = set(obj2.keys())

            # Check for missing keys
            # Sometimes flatbuffers will omit fields/keys when the value is empty
            # Therefore if a key is missing, we need to check that the value is empty for the object that has the key
            if keys1 != keys2:
                missing_in_2 = keys1 - keys2
                missing_in_1 = keys2 - keys1

                # For keys in first but not second, check they have empty values
                for key in missing_in_2:
                    value = obj1[key]
                    if not _is_empty_container(value):
                        differences.append(
                            f"{path}.{key}: Key exists in first but not second, and has non-empty value: {value}"
                        )

                # For keys in second but not first, check they have empty values
                for key in missing_in_1:
                    value = obj2[key]
                    if not _is_empty_container(value):
                        differences.append(
                            f"{path}.{key}: Key exists in second but not first, and has non-empty value: {value}"
                        )

            # Compare common keys
            for key in keys1 & keys2:
                # Skip the erisc_l1_unreserved_base field
                # This field will be different on multi host because of extra fabric firmware overhead
                if key == "erisc_l1_unreserved_base":
                    continue

                new_path = f"{path}.{key}" if path else key
                _deep_compare(obj1[key], obj2[key], new_path, differences)

        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                differences.append(
                    f"{path}: List length mismatch - {len(obj1)} vs {len(obj2)}"
                )
                return

            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                new_path = f"{path}[{i}]"
                _deep_compare(item1, item2, new_path, differences)

        else:
            # Compare primitive values
            if obj1 != obj2:
                differences.append(f"{path}: Value mismatch - {obj1} vs {obj2}")

    differences = []
    _deep_compare(desc1, desc2, "", differences)

    are_equal = len(differences) == 0

    return are_equal, differences


def test_system_desc(request):
    system_desc_local = subprocess_get_system_descriptor(request)

    launch_distributed_runtime()

    system_desc_distributed = ttrt.runtime.get_current_system_desc()
    assert system_desc_distributed is not None

    shutdown_distributed_runtime()

    are_equal, differences = compare_system_descriptors(
        json.loads(system_desc_local.as_json()),
        json.loads(system_desc_distributed.as_json()),
    )

    assert are_equal, f"System descriptor mismatch with differences: {differences}"


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
        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
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
            _,
        ) = test_runner.get_inputs_and_golden(submesh1, borrow=False)
        (
            inputs_runtime_with_layout_submesh2,
            golden2,
            _,
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


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((177, 211), torch.float32),
        ((32, 64), torch.bfloat16),
        ((100, 50), torch.bfloat16),
        ((10, 20), torch.int32),
        ((2, 3, 4), torch.bfloat16),
        ((1, 3, 224, 224), torch.bfloat16),
    ],
)
def test_getTensorDesc(shape, dtype):
    launch_distributed_runtime()
    if dtype in [torch.int8, torch.uint8, torch.int32]:
        reference_torch_tensor = torch.randint(-10, 10, shape, dtype=dtype)
    else:
        reference_torch_tensor = torch.randn(shape, dtype=dtype)

    tensor = get_runtime_tensor_from_torch(
        reference_torch_tensor, storage=Storage.Owned
    )

    tensor_desc = tensor.get_tensor_desc()

    # Assert tensor descriptor properties match the reference tensor
    assert tensor_desc.shape == list(reference_torch_tensor.shape)
    expected_runtime_dtype = Binary.Program.to_data_type(reference_torch_tensor.dtype)
    assert tensor_desc.dtype == expected_runtime_dtype
    assert tensor_desc.item_size == reference_torch_tensor.element_size()

    # Physical volume is typically 0 for host tensors (not on device)
    assert tensor_desc.physical_volume == 0

    shutdown_distributed_runtime()


@pytest.mark.parametrize("enable_program_cache", [True, False])
def test_isProgramCacheEnabled(enable_program_cache):
    launch_distributed_runtime()

    with DeviceContext(
        mesh_shape=[1, 8], enable_program_cache=enable_program_cache
    ) as device:
        assert device.is_program_cache_enabled() == enable_program_cache
        # It is currently not possible to inspect the contents of program cache from tt-mlir runtime
        # so this test just checks that this function doesn't throw
        device.clear_program_cache()

    shutdown_distributed_runtime()


layout_funcs = [
    ttrt.runtime.test.get_dram_interleaved_tile_layout,
    ttrt.runtime.test.get_dram_interleaved_row_major_layout,
    ttrt.runtime.test.get_host_row_major_layout,
]


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((177, 211), torch.float32),
        ((32, 64), torch.bfloat16),
    ],
)
@pytest.mark.parametrize("layout_func", layout_funcs)
def test_hasLayout(shape, dtype, layout_func):
    reference_torch_tensor = torch.zeros(shape, dtype=dtype)

    tensor = get_runtime_tensor_from_torch(
        reference_torch_tensor, storage=Storage.Owned
    )
    runtime_dtype = Binary.Program.to_data_type(dtype)

    device_layout = layout_func(runtime_dtype)
    wrong_layout_funcs = [f for f in layout_funcs if f is not layout_func]
    wrong_layouts = [
        wrong_layout_func(runtime_dtype) for wrong_layout_func in wrong_layout_funcs
    ]

    with DeviceContext(mesh_shape=[1, 1]) as device:
        device_tensor = ttrt.runtime.to_layout(tensor, device, device_layout)
        assert device_tensor.has_layout(device_layout)
        for wrong_layout in wrong_layouts:
            assert not device_tensor.has_layout(wrong_layout)
