# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from ..utils import TT_MLIR_HOME, Helper, DeviceContext, assert_pcc


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
