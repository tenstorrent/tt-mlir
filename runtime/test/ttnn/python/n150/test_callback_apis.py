# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch.nn.functional as F
from typing import Callable
import ttrt
import ttrt.runtime
from ttrt.common.util import *
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
    assert_pcc,
    get_torch_output_container,
    get_to_layout_inputs,
    get_runtime_tensor_from_torch,
    get_torch_inputs,
)

FLATBUFFER_BASE_PATHS_AND_PROGRAMS = {
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/trace/Output/mnist_linear_logits.mlir.tmp.ttnn": 5,
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/tensor_manipulation/Output/linear.mlir.tmp.ttnn": 4,
}


# ****** TEMP


def get_torch_tensor(tensor: ttrt.runtime.Tensor):
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    if rt_dtype is not ttrt.runtime.DataType.Float32:
        raise ValueError(f"Unsupported data type: {rt_dtype}")
    dtype = torch.float32
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    torch_tensor = torch_tensor.reshape(shape)
    return torch_tensor


def update_device_tensor(program_context, tensor_ref, dst_tensor, src_tensor):
    data_ptr = src_tensor.data_ptr()
    shape = dst_tensor.get_shape()
    stride = dst_tensor.get_stride()
    dtype = dst_tensor.get_dtype()
    size = torch.numel(src_tensor)
    tensor = ttrt.runtime.create_owned_host_tensor(data_ptr, shape, stride, size, dtype)
    ttrt.runtime.update_tensor_in_pool(program_context, tensor_ref, tensor)


def pre_op_callback(binary, programContext, opContext):
    pass


def get_output_tensor_callback(binary, programContext, opContext):
    debug_op_str = ttrt.runtime.get_op_debug_str(opContext)

    if "ttnn.linear" not in debug_op_str:
        return

    output_tensor_map = ttrt.runtime.get_op_output_tensor(opContext, programContext)
    for index, tensor in output_tensor_map.items():
        print(
            f"Output tensor at index {index}: shape={tensor.get_shape()}, dtype={tensor.get_dtype()}"
        )


def get_output_ref_callback(binary, programContext, opContext):
    debug_op_str = ttrt.runtime.get_op_debug_str(opContext)

    if "ttnn.linear" not in debug_op_str:
        return

    output_ref_map = ttrt.runtime.get_op_output_tensor_ref(opContext)
    for index, tensor_ref in output_ref_map.items():
        print(f"Output tensor ref at index {index}: ref={tensor_ref}")


def get_input_ref_callback(binary, programContext, opContext):
    debug_op_str = ttrt.runtime.get_op_debug_str(opContext)

    if "ttnn.linear" not in debug_op_str:
        return

    input_ref_map = ttrt.runtime.get_op_input_tensor_ref(opContext)
    for index, tensor_ref in input_ref_map.items():
        print(f"Input tensor ref at index {index}: ref={tensor_ref}")


def retrieve_tensor_from_pool_callback(binary, programContext, opContext):
    pass


def update_tensor_in_pool_callback(binary, programContext, opContext):
    debug_op_str = ttrt.runtime.get_op_debug_str(opContext)

    if "ttnn.linear" not in debug_op_str:
        return

    tensor_ref: ttrt.runtime.TensorRef = ttrt.runtime.get_op_output_ref(
        opContext, programContext
    )
    if tensor_ref is None:
        return

    tensor: ttrt.runtime.Tensor = ttrt.runtime.retrieve_tensor_from_pool(
        programContext, tensor_ref
    )
    if tensor is None:
        return

    torch_tensor = get_torch_tensor(tensor)

    print(torch_tensor)
    # For linear operation with all-ones inputs: input(10x10) @ weight(10x10) + bias(10)
    # Each element will be 10 (from matmul) + 1 (from bias) = 11
    assert torch.allclose(torch_tensor, torch.ones_like(torch_tensor) * 11)

    update_device_tensor(
        programContext, tensor_ref, tensor, torch.ones_like(torch_tensor)
    )


binary_paths = [
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/consteval/Output/binary_ops.mlir.tmp.ttnn"
]
callback_fns = [
    # get_output_tensor_callback,
    # get_output_ref_callback,
    # get_input_ref_callback,
    # retrieve_tensor_from_pool_callback,
    update_tensor_in_pool_callback,
]


@pytest.mark.parametrize("binary_path", binary_paths)
@pytest.mark.parametrize("callback_fn", callback_fns)
def test_callbacks(binary_path: str, callback_fn: Callable, helper: Helper, request):
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_runner = ProgramTestRunner(helper.binary, 0)
    rand_inputs_torch = get_torch_inputs(test_runner.program)
    inputs_torch = [torch.ones_like(input) for input in rand_inputs_torch]

    hooks = ttrt.runtime.DebugHooks.get(pre_op_callback, callback_fn)

    if hooks is None:
        # Hooks are not enabled
        return

    with DeviceContext(mesh_shape=[1, 1]) as device:
        runtime_inputs = [
            get_runtime_tensor_from_torch(input) for input in inputs_torch
        ]
        inputs_runtime_with_layout = get_to_layout_inputs(
            device, runtime_inputs, helper.binary, 0
        )
        output_torch = get_torch_output_container(test_runner.program)
        output = test_runner.run_program(device, inputs_runtime_with_layout)
        ttrt.runtime.memcpy(output_torch.data_ptr(), output)
        # assert torch.allclose(output_torch, torch.ones_like(output_torch) * 2)

    ttrt.runtime.unregister_hooks()
    helper.teardown()
