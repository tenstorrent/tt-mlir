# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from functools import partial
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from ttrt.common.callback import CallbackRuntimeConfig
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    assert_pcc,
    get_torch_inputs,
    get_runtime_tensor_from_torch,
    get_to_layout_inputs,
)

FLATBUFFER_BASE_PATH = f"{TT_MLIR_HOME}/build/test/ttmlir/Silicon/TTNN/n300/perf/Output"


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tensor_buffer_api(shape, dtype):
    torch_tensor = torch.randn(shape, dtype=dtype)
    runtime_dtype = Binary.Program.to_data_type(dtype)
    rt_tensor = ttrt.runtime.create_tensor(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        list(torch_tensor.stride()),
        torch_tensor.element_size(),
        runtime_dtype,
    )
    rt_shape = rt_tensor.get_shape()
    rt_stride = rt_tensor.get_stride()
    rt_elem_size = rt_tensor.get_element_size()
    rt_vol = rt_tensor.get_volume()
    rt_dtype = ttrt_datatype_to_torch_dtype(rt_tensor.get_dtype())
    rt_bytes = rt_tensor.get_data_buffer()
    rt_desc = rt_tensor.get_tensor_desc()

    # Tests to make sure the binding of `TensorDesc` works. Might belong in its own test?
    assert rt_desc.shape == rt_shape
    assert rt_desc.stride == rt_stride
    assert ttrt_datatype_to_torch_dtype(rt_desc.dtype) == rt_dtype
    assert rt_desc.item_size == rt_elem_size

    # Various tests that the no underlying stuff has changed over the pybind boundary
    assert list(rt_shape) == list(shape)
    assert list(rt_stride) == list(torch_tensor.stride())
    assert rt_elem_size == torch_tensor.element_size()
    assert rt_vol == torch_tensor.numel()
    assert rt_dtype == torch_tensor.dtype
    assert len(rt_bytes) == rt_vol * rt_elem_size
    reconstructed_tensor = torch.frombuffer(rt_bytes, dtype=rt_dtype).reshape(rt_shape)
    assert torch.equal(torch_tensor, reconstructed_tensor)


@pytest.mark.parametrize("should_retain", [True, False])
def test_tensor_retain_api(helper: Helper, should_retain, request):
    helper.initialize(request.node.name)
    helper.check_constraints()
    torch_tensor = torch.randn((64, 128))
    runtime_tensor = get_runtime_tensor_from_torch(torch_tensor)
    runtime_tensor.set_retain(should_retain)
    assert runtime_tensor.get_retain() == should_retain


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
    with DeviceContext(mesh_shape=[1, 1]) as device:
        device_tensor = ttrt.runtime.to_layout(
            runtime_input_tensor, device, device_layout
        )
        host_tensor = ttrt.runtime.to_layout(device_tensor, device, host_layout)
        ttrt.runtime.deallocate_tensor(device_tensor, force=True)
        ttrt.runtime.memcpy(runtime_output_tensor, host_tensor)
        ttrt.runtime.deallocate_tensor(host_tensor, force=True)

    assert_pcc(torch_input_tensor, torch_result_tensor, threshold=0.99)
    helper.teardown()


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_memcpy_to_pointer(helper: Helper, shape, dtype, request):
    helper.initialize(request.node.name)
    helper.check_constraints()
    runtime_dtype = Binary.Program.to_data_type(dtype)
    torch_result_tensor = torch.zeros(shape, dtype=dtype)

    # Device to host
    torch_input_tensor = torch.randn(shape, dtype=dtype)
    runtime_input_tensor = ttrt.runtime.create_tensor(
        torch_input_tensor.data_ptr(),
        list(torch_input_tensor.shape),
        list(torch_input_tensor.stride()),
        torch_input_tensor.element_size(),
        runtime_dtype,
    )
    device_layout = ttrt.runtime.testing.get_dram_interleaved_row_major_layout(
        runtime_dtype
    )
    with DeviceContext(mesh_shape=[1, 1]) as device:
        device_tensor = ttrt.runtime.to_layout(
            runtime_input_tensor, device, device_layout
        )
        ttrt.runtime.memcpy(torch_result_tensor.data_ptr(), device_tensor)
        ttrt.runtime.deallocate_tensor(device_tensor, force=True)

    assert_pcc(torch_input_tensor, torch_result_tensor, threshold=0.99)

    # Host to host
    torch_input_tensor2 = torch.randn(shape, dtype=dtype)
    host_tensor = ttrt.runtime.create_tensor(
        torch_input_tensor2.data_ptr(),
        list(torch_input_tensor2.shape),
        list(torch_input_tensor2.stride()),
        torch_input_tensor2.element_size(),
        runtime_dtype,
    )
    ttrt.runtime.memcpy(torch_result_tensor.data_ptr(), host_tensor)
    assert_pcc(torch_input_tensor2, torch_result_tensor, threshold=0.99)
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
    with DeviceContext(mesh_shape=[1, 1]) as device:
        device_tensor = ttrt.runtime.create_empty_tensor(
            device,
            device_layout,
            list(torch_input_tensor.shape),
            list(torch_input_tensor.stride()),
            torch_input_tensor.element_size(),
        )
        # Copy from host to device container
        ttrt.runtime.memcpy(device_tensor, runtime_input_tensor)
        # Copy from device to host
        ttrt.runtime.memcpy(runtime_output_tensor, device_tensor)
        ttrt.runtime.deallocate_tensor(device_tensor, force=True)
    assert_pcc(torch_input_tensor, torch_result_tensor, threshold=0.99)
    helper.teardown()


def test_set_program_cache(helper):
    num_devices = ttrt.runtime.get_num_available_devices()
    with DeviceContext(
        mesh_shape=[1, num_devices], enable_program_cache=False
    ) as device:
        assert (
            ttrt.runtime.testing.is_program_cache_enabled(device) == False
        ), "Expected program cache to be disabled"

    with DeviceContext(
        mesh_shape=[1, num_devices], enable_program_cache=True
    ) as device:
        assert (
            ttrt.runtime.testing.is_program_cache_enabled(device) == True
        ), "Expected program cache to be enabled"


@pytest.mark.parametrize(
    "runtime",
    [ttrt.runtime.DeviceRuntime.TTNN, ttrt.runtime.DeviceRuntime.TTMetal],
    ids=["ttnn", "ttmetal"],
)
@pytest.mark.parametrize(
    "dispatch_core_type",
    [None, ttrt.runtime.DispatchCoreType.ETH, ttrt.runtime.DispatchCoreType.WORKER],
    ids=["no_dispatch_core", "eth_dispatch_core", "worker_dispatch_core"],
)
@pytest.mark.parametrize("with_device", [False, True], ids=["no_device", "with_device"])
def test_get_system_desc(runtime, dispatch_core_type, with_device):
    ttrt.runtime.set_current_runtime(runtime)
    num_devices = ttrt.runtime.get_num_available_devices()

    if with_device:
        with DeviceContext(mesh_shape=[1, num_devices]) as device:
            system_desc, device_ids = ttrt.runtime.get_current_system_desc(
                dispatch_core_type, device
            )

    else:
        system_desc, device_ids = ttrt.runtime.get_current_system_desc(
            dispatch_core_type
        )

        assert (
            len(device_ids) == num_devices
        ), f"Expected {num_devices} device IDs, got {len(device_ids)}"

    assert system_desc is not None, "System descriptor should exist"
    assert device_ids is not None, "Device IDs should exist"

    sorted_device_ids = sorted(device_ids)
    assert (
        device_ids == sorted_device_ids
    ), f"Expected device IDs {sorted_device_ids}, got {device_ids}"


def pre_op_callback(callback_runtime_config, binary, program_context, op_context):
    # Testing apis in pre op callback function

    callback_runtime_config.save_intermediates(program_context, op_context)
    intermeds = callback_runtime_config.intermediates
    in_tensor_ids = ttrt.runtime.get_input_tensor_ids(program_context)
    out_tensor_ids = ttrt.runtime.get_output_tensor_ids(program_context)
    in_tensors = ttrt.runtime.get_input_tensors(program_context)
    out_tensors = ttrt.runtime.get_output_tensors(program_context)
    intermed_in_tensor_ids = ttrt.runtime.get_intermediate_input_tensor_ids(op_context)
    intermed_out_tensor_id = ttrt.runtime.get_intermediate_output_tensor_id(op_context)

    op_intermediate_tensors = ttrt.runtime.get_intermediate_input_tensors(
        op_context, program_context
    )
    for intermed_in_tensor_id in intermed_in_tensor_ids:
        if ttrt.runtime.is_tensor_live(program_context, intermed_in_tensor_id):
            op_intermediate_tensor_get = ttrt.runtime.get_tensor(
                program_context, intermed_out_tensor_id
            )


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_callback(callback_runtime_config, binary, program_context, op_context):
    # Testing apis in post op callback function

    callback_runtime_config.save_intermediates(program_context, op_context)
    intermeds = callback_runtime_config.intermediates
    in_tensor_ids = ttrt.runtime.get_input_tensor_ids(program_context)
    out_tensor_ids = ttrt.runtime.get_output_tensor_ids(program_context)
    in_tensors = ttrt.runtime.get_input_tensors(program_context)
    out_tensors = ttrt.runtime.get_output_tensors(program_context)
    intermed_in_tensor_ids = ttrt.runtime.get_intermediate_input_tensor_ids(op_context)
    intermed_out_tensor_id = ttrt.runtime.get_intermediate_output_tensor_id(op_context)

    if ttrt.runtime.is_tensor_live(program_context, intermed_out_tensor_id):
        op_intermediate_tensor = ttrt.runtime.get_intermediate_output_tensor(
            op_context, program_context
        )
        op_intermediate_tensor_get = ttrt.runtime.get_tensor(
            program_context, intermed_out_tensor_id
        )


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)


# NOTE: All callback API functions are run, but verification is not implemented yet
def test_callback_apis(
    helper: Helper,
    request,
):
    """Test that callback APIs work as expected."""
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "all_gather.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    num_devices = ttrt.runtime.get_num_available_devices()
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    # Set the current runtime to TTNN, Callback APIs are not supported in TTMetal
    ttrt.runtime.set_current_runtime(ttrt.runtime.DeviceRuntime.TTNN)
    program: Binary.Program = helper.binary.get_program(0)

    torch_inputs = get_torch_inputs(program)
    runtime_inputs = [
        get_runtime_tensor_from_torch(torch_input) for torch_input in torch_inputs
    ]

    with DeviceContext(mesh_shape=[1, num_devices]) as parent_mesh:
        runtime_inputs_with_layouts = get_to_layout_inputs(
            parent_mesh, runtime_inputs, helper.binary, 0
        )
        # Set up pre and post op callback hooks
        callback_config = [parent_mesh, "", 0.99, 1e-08, 1e-05, False, True]
        pre_op_callback_runtime_config = CallbackRuntimeConfig(*callback_config)
        post_op_callback_runtime_config = CallbackRuntimeConfig(*callback_config)
        callback_env = ttrt.runtime.DebugHooks.get(
            pre_op_get_callback_fn(pre_op_callback_runtime_config),
            post_op_get_callback_fn(post_op_callback_runtime_config),
        )

        # Perform submit operation
        output = ttrt.runtime.submit(
            parent_mesh, helper.binary.fbb, 0, runtime_inputs_with_layouts
        )[0]
        output_host = ttrt.runtime.to_host(output, untilize=True)[0]

    helper.teardown()
