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
    assert_pcc,
    get_torch_inputs,
    get_runtime_tensor_from_torch,
    get_torch_output_container,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/runtime_stitching/Output"
)


def test_runtime_stitching_eltwise_binary_op_chain(helper: Helper, request):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "eltwise_binary_op_chain.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    first_program: Binary.Program = helper.binary.get_program(0)
    assert first_program.num_inputs() == 2
    inputs_torch = get_torch_inputs(first_program)
    inputs_runtime = [get_runtime_tensor_from_torch(input) for input in inputs_torch]

    input_layouts = [
        ttrt.runtime.get_layout(
            executable=helper.binary.fbb, program_index=0, input_index=i
        )
        for i in range(len(inputs_runtime))
    ]

    program_indices = list(range(helper.binary.get_num_programs()))
    last_program: Binary.Program = helper.binary.get_program(program_indices[-1])
    torch_result_tensor = get_torch_output_container(last_program)

    activations, weights = inputs_runtime
    activations_layout, weights_layout = input_layouts
    with DeviceContext(mesh_shape=[1, 1]) as device:
        activations = ttrt.runtime.to_layout(activations, device, activations_layout)
        weights = ttrt.runtime.to_layout(weights, device, weights_layout)
        weights.set_retain(True)
        for program_index in program_indices:
            program = helper.binary.get_program(program_index)
            assert program.num_inputs() == 2 and program.num_outputs() == 1
            inputs = [activations, weights]
            outputs = ttrt.runtime.submit(
                device, helper.binary.fbb, program_index, inputs
            )
            activations = ttrt.runtime.to_layout(outputs[0], device, activations_layout)
            ttrt.runtime.deallocate_tensor(outputs[0])
        final_result = ttrt.runtime.to_host(activations, untilize=True)[0]
        ttrt.runtime.memcpy(torch_result_tensor.data_ptr(), final_result)
        ttrt.runtime.deallocate_tensor(activations, force=True)
        ttrt.runtime.deallocate_tensor(weights, force=True)
        ttrt.runtime.deallocate_tensor(final_result, force=True)

    golden = (
        (inputs_torch[0] + inputs_torch[1]).mul(inputs_torch[1]).sub(inputs_torch[1])
    )
    assert_pcc(golden, torch_result_tensor, threshold=0.99)
    helper.teardown()
