# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from .constants import FLATBUFFER_BASE_PATH

from ..utils import (
    Helper,
    DeviceContext,
    assert_pcc,
    get_to_layout_inputs,
    get_runtime_tensor_from_torch,
)

MESH_SHAPE = [1, 2]


def get_input_spec(program, input_index):
    """Extract shape and dtype from program input specification."""
    program_input = program.inputs[input_index]
    shape = program_input["desc"]["shape"]
    dtype = Binary.Program.from_data_type(
        program_input["desc"]["layout"]["memory_desc"]["data_type"]
    )
    return shape, dtype


def create_multi_device_input(shape, dtype):
    """Create a multi-device host tensor with random data per shard."""
    runtime_dtype = Binary.Program.to_data_type(dtype)
    num_devices = MESH_SHAPE[0] * MESH_SHAPE[1]
    torch_shards = [torch.randn(shape, dtype=dtype) for _ in range(num_devices)]

    runtime_tensor = ttrt.runtime.create_multi_device_host_tensor(
        [shard.data_ptr() for shard in torch_shards],
        list(shape),
        list(torch_shards[0].stride()),
        torch_shards[0].element_size(),
        runtime_dtype,
        {},
        MESH_SHAPE,
    )
    return runtime_tensor, torch_shards


def create_single_device_input(shape, dtype):
    """Create a single-device host tensor with random data."""
    torch_tensor = torch.randn(shape, dtype=dtype)
    runtime_tensor = get_runtime_tensor_from_torch(torch_tensor)
    return runtime_tensor, torch_tensor


def device_tensor_to_torch(runtime_tensor):
    """Convert a runtime device tensor to a torch tensor."""
    host_tensor = ttrt.runtime.to_host(runtime_tensor, untilize=True)
    assert len(host_tensor) == 1

    shape = host_tensor[0].get_shape()
    dtype = ttrt_datatype_to_torch_dtype(host_tensor[0].get_dtype())
    torch_tensor = torch.zeros(shape, dtype=dtype)
    ttrt.runtime.memcpy(torch_tensor.data_ptr(), host_tensor[0])
    return torch_tensor


def run_and_verify(helper, mesh_device, runtime_inputs, expected_output_shards):
    """Submit program and verify output shards match expected values."""
    runtime_inputs_with_layout = get_to_layout_inputs(
        mesh_device, runtime_inputs, helper.binary, 0
    )

    output = ttrt.runtime.submit(
        mesh_device, helper.binary.fbb, 0, runtime_inputs_with_layout
    )[0]

    output_device_tensors = ttrt.runtime.get_device_tensors(output)
    assert len(output_device_tensors) == len(expected_output_shards)

    for i, output_tensor in enumerate(output_device_tensors):
        torch_output = device_tensor_to_torch(output_tensor)
        assert_pcc(expected_output_shards[i], torch_output, threshold=0.99)

    ttrt.runtime.deallocate_tensor(output, force=True)


def test_cpu_hoisted_add(helper: Helper, request):
    """Test CPU-hoisted add with both inputs sharded."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "cpu_hoisted_add.mlir.tmp.ttnn")
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_shards = create_multi_device_input(shape, dtype)
    input1, input1_shards = create_multi_device_input(shape, dtype)

    expected = [torch.add(input0_shards[i], input1_shards[i]) for i in range(2)]

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify(helper, mesh_device, [input0, input1], expected)

    helper.teardown()


def test_cpu_hoisted_add_mixed_inputs(helper: Helper, request):
    """Test CPU-hoisted add with one sharded and one non-sharded input."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "cpu_hoisted_add.mlir.tmp.ttnn")
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_shards = create_multi_device_input(shape, dtype)
    input1, input1_torch = create_single_device_input(shape, dtype)

    # Non-sharded input is broadcast to all shards.
    expected = [torch.add(input0_shards[i], input1_torch) for i in range(2)]

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify(helper, mesh_device, [input0, input1], expected)

    helper.teardown()
