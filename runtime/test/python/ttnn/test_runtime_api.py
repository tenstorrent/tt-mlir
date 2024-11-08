# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from utils import TT_MLIR_HOME, Helper, DeviceContext, assert_pcc


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_to_layout(helper: Helper, shape, dtype, request):
    helper.initialize(request.node.name)
    helper.check_constraints()
    torch_input_tensor = torch.randn(shape, dtype=dtype)
    torch_result_tensor = torch.zeros(shape, dtype=dtype)
    runtime_dtype = Binary.Program.to_data_type(dtype)
    runtime_input_tensor = ttrt.runtime.create_tensor(
        torch_input_tensor.data_ptr(),
        list(torch_input_tensor.shape),
        list(torch_input_tensor.stride()),
        torch_input_tensor.element_size(),
        runtime_dtype,
    )
    runtime_output_tensor = ttrt.runtime.create_tensor(
        torch_result_tensor.data_ptr(),
        list(torch_result_tensor.shape),
        list(torch_result_tensor.stride()),
        torch_result_tensor.element_size(),
        runtime_dtype,
    )
    device_layout = ttrt.runtime.testing.get_dram_interleaved_tile_layout(runtime_dtype)
    host_layout = ttrt.runtime.testing.get_host_row_major_layout(runtime_dtype)
    with DeviceContext([helper.query.device_ids[0]]) as device:
        device_tensor = ttrt.runtime.to_layout(
            runtime_input_tensor, device, device_layout
        )
        host_tensor = ttrt.runtime.to_layout(device_tensor, device, host_layout)
        ttrt.runtime.deallocate_tensor(device_tensor, force=True)
        ttrt.runtime.memcpy(runtime_output_tensor, host_tensor)
        ttrt.runtime.deallocate_tensor(host_tensor, force=True)

    lambda: assert_pcc(torch_input_tensor, torch_result_tensor, threshold=0.999)
    helper.teardown()


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_create_tensor_memcpy(helper: Helper, shape, dtype, request):
    helper.initialize(request.node.name)
    helper.check_constraints()
    torch_input_tensor = torch.randn(shape, dtype=dtype)
    torch_result_tensor = torch.zeros(shape, dtype=dtype)
    runtime_dtype = Binary.Program.to_data_type(dtype)
    runtime_input_tensor = ttrt.runtime.create_tensor(
        torch_input_tensor.data_ptr(),
        list(torch_input_tensor.shape),
        list(torch_input_tensor.stride()),
        torch_input_tensor.element_size(),
        runtime_dtype,
    )
    runtime_output_tensor = ttrt.runtime.create_tensor(
        torch_result_tensor.data_ptr(),
        list(torch_result_tensor.shape),
        list(torch_result_tensor.stride()),
        torch_result_tensor.element_size(),
        runtime_dtype,
    )
    device_layout = ttrt.runtime.testing.get_dram_interleaved_row_major_layout(
        runtime_dtype
    )
    with DeviceContext([helper.query.device_ids[0]]) as device:
        device_tensor = ttrt.runtime.create_empty_tensor(
            device,
            device_layout,
            list(torch_input_tensor.shape),
            list(torch_input_tensor.stride()),
            torch_input_tensor.element_size(),
        )
        ttrt.runtime.memcpy(device_tensor, runtime_input_tensor)
        host_tensor = ttrt.runtime.to_host(device_tensor, untilize=True)
        ttrt.runtime.deallocate_tensor(device_tensor, force=True)
        ttrt.runtime.memcpy(runtime_output_tensor, host_tensor)
        ttrt.runtime.deallocate_tensor(host_tensor, force=True)
    lambda: assert_pcc(torch_input_tensor, torch_result_tensor, threshold=0.999)
    helper.teardown()


def test_runtime_stitching_eltwise_binary_op_chain(helper: Helper, request):
    binary_path = f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/runtime_stitching/Output/eltwise_binary_op_chain.mlir.tmp.ttnn"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()
    first_program: Binary.Program = helper.binary.get_program(0)
    assert first_program.num_inputs() == 2
    inputs_torch = []
    inputs_runtime = []
    input_layouts = []
    for i in first_program.program["inputs"]:
        torch_tensor = torch.randn(
            i["desc"]["shape"],
            dtype=Binary.Program.from_data_type(
                i["desc"]["layout"]["memory_desc"]["data_type"]
            ),
        )
        runtime_dtype = Binary.Program.to_data_type(torch_tensor.dtype)
        inputs_torch.append(torch_tensor)
        runtime_tensor = ttrt.runtime.create_tensor(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            list(torch_tensor.stride()),
            torch_tensor.element_size(),
            runtime_dtype,
        )
        inputs_runtime.append(runtime_tensor)
        input_layouts.append(
            ttrt.runtime.testing.get_dram_interleaved_row_major_layout(runtime_dtype)
        )

    activations, weights = inputs_runtime
    activations_layout, weights_layout = input_layouts
    with DeviceContext([helper.query.device_ids[0]]) as device:
        activations = ttrt.runtime.to_layout(activations, device, activations_layout)
        weights = ttrt.runtime.to_layout(weights, device, weights_layout)
        program_indices = list(range(helper.binary.get_num_programs()))
        for program_index in program_indices:
            program = helper.binary.get_program(program_index)
            assert program.num_inputs() == 2 and program.num_outputs() == 1
            outputs = ttrt.runtime.submit(
                device, helper.binary.fbb, program_index, [activations, weights]
            )
            activations = ttrt.runtime.to_layout(outputs[0], device, activations_layout)
            ttrt.runtime.deallocate_tensor(outputs[0], force=True)
        activations = ttrt.runtime.to_host(activations, untilize=True)
        ttrt.runtime.deallocate_tensor(weights, force=True)

    last_program: Binary.Program = helper.binary.get_program(program_indices[-1])
    torch_result_tensor = torch.randn(
        last_program.program["outputs"][0]["desc"]["shape"],
        dtype=Binary.Program.from_data_type(
            last_program.program["outputs"][0]["desc"]["layout"]["memory_desc"][
                "data_type"
            ]
        ),
    )
    runtime_result_tensor = ttrt.runtime.create_tensor(
        torch_result_tensor.data_ptr(),
        list(torch_result_tensor.shape),
        list(torch_result_tensor.stride()),
        torch_result_tensor.element_size(),
        Binary.Program.to_data_type(torch_result_tensor.dtype),
    )
    ttrt.runtime.memcpy(runtime_result_tensor, activations)
    golden = (
        (inputs_torch[0] + inputs_torch[1]).mul(inputs_torch[1]).sub(inputs_torch[1])
    )
    assert_pcc(golden, torch_result_tensor, threshold=0.999), program_index
    helper.teardown()
