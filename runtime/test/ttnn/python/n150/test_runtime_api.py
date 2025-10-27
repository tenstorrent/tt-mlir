# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from ..utils import Helper, DeviceContext, assert_pcc, get_runtime_tensor_from_torch


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tensor_buffer_api(shape, dtype):
    torch_tensor = torch.randn(shape, dtype=dtype)
    runtime_dtype = Binary.Program.to_data_type(dtype)
    rt_tensor = ttrt.runtime.create_borrowed_host_tensor(
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


@pytest.mark.parametrize("shape", [(64, 128), (4, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_to_layout(helper: Helper, shape, dtype, request):
    helper.initialize(request.node.name)
    helper.check_constraints()
    torch_input_tensor = torch.randn(shape, dtype=dtype)
    torch_result_tensor = torch.zeros(shape, dtype=dtype)
    runtime_dtype = Binary.Program.to_data_type(dtype)
    runtime_input_tensor = ttrt.runtime.create_borrowed_host_tensor(
        torch_input_tensor.data_ptr(),
        list(torch_input_tensor.shape),
        list(torch_input_tensor.stride()),
        torch_input_tensor.element_size(),
        runtime_dtype,
    )
    runtime_output_tensor = ttrt.runtime.create_borrowed_host_tensor(
        torch_result_tensor.data_ptr(),
        list(torch_result_tensor.shape),
        list(torch_result_tensor.stride()),
        torch_result_tensor.element_size(),
        runtime_dtype,
    )
    device_layout = ttrt.runtime.test.get_dram_interleaved_tile_layout(runtime_dtype)
    host_layout = ttrt.runtime.test.get_host_row_major_layout(runtime_dtype)

    input_tensor_logical_volume = runtime_input_tensor.get_volume()
    with DeviceContext(mesh_shape=[1, 1]) as device:
        device_tensor = ttrt.runtime.to_layout(
            runtime_input_tensor, device, device_layout
        )
        assert device_tensor.get_logical_volume() == input_tensor_logical_volume

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
    runtime_input_tensor = ttrt.runtime.create_borrowed_host_tensor(
        torch_input_tensor.data_ptr(),
        list(torch_input_tensor.shape),
        list(torch_input_tensor.stride()),
        torch_input_tensor.element_size(),
        runtime_dtype,
    )
    device_layout = ttrt.runtime.test.get_dram_interleaved_row_major_layout(
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
    host_tensor = ttrt.runtime.create_borrowed_host_tensor(
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
    runtime_input_tensor = ttrt.runtime.create_borrowed_host_tensor(
        torch_input_tensor.data_ptr(),
        list(torch_input_tensor.shape),
        list(torch_input_tensor.stride()),
        torch_input_tensor.element_size(),
        runtime_dtype,
    )
    runtime_output_tensor = ttrt.runtime.create_borrowed_host_tensor(
        torch_result_tensor.data_ptr(),
        list(torch_result_tensor.shape),
        list(torch_result_tensor.stride()),
        torch_result_tensor.element_size(),
        runtime_dtype,
    )
    device_layout = ttrt.runtime.test.get_dram_interleaved_row_major_layout(
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
    ttrt.runtime.set_current_device_runtime(runtime)
    num_devices = ttrt.runtime.get_num_available_devices()

    if with_device:
        with DeviceContext(mesh_shape=[1, num_devices]) as device:
            system_desc = ttrt.runtime.get_current_system_desc(
                dispatch_core_type, device
            )

    else:
        system_desc = ttrt.runtime.get_current_system_desc(dispatch_core_type)

    assert system_desc is not None, "System descriptor should exist"


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.int64, torch.uint64, torch.int16, torch.int8, torch.bool],
)
def test_create_owned_tensor_with_unsupported_data_type(dtype):
    ttrt.runtime.set_current_device_runtime(ttrt.runtime.DeviceRuntime.TTNN)
    torch_input_tensor = (127 * torch.rand((64, 128))).to(dtype)
    runtime_dtype = Binary.Program.to_data_type(dtype)
    runtime_input_tensor = ttrt.runtime.create_owned_host_tensor(
        torch_input_tensor.data_ptr(),
        list(torch_input_tensor.shape),
        list(torch_input_tensor.stride()),
        torch_input_tensor.element_size(),
        runtime_dtype,
    )

    torch_output_tensor = torch.zeros_like(torch_input_tensor)
    ttrt.runtime.memcpy(
        torch_output_tensor.data_ptr(),
        runtime_input_tensor,
        Binary.Program.to_data_type(dtype),
    )
    assert torch.all(torch_output_tensor == torch_input_tensor)


@pytest.mark.parametrize("num_loops", [64])
def test_unblocking_to_host(num_loops):
    ttrt.runtime.set_current_device_runtime(ttrt.runtime.DeviceRuntime.TTNN)
    dtype = torch.bfloat16
    torch_input_tensor = torch.randn((256, 784, 892), dtype=dtype)
    runtime_dtype = Binary.Program.to_data_type(dtype)
    device_layout = ttrt.runtime.test.get_dram_interleaved_tile_layout(runtime_dtype)
    runtime_dtype = Binary.Program.to_data_type(dtype)
    runtime_input_tensor = ttrt.runtime.create_owned_host_tensor(
        torch_input_tensor.data_ptr(),
        list(torch_input_tensor.shape),
        list(torch_input_tensor.stride()),
        torch_input_tensor.element_size(),
        runtime_dtype,
    )

    with DeviceContext(mesh_shape=[1, 1]) as device:
        for _ in range(num_loops):
            runtime_device_tensor = ttrt.runtime.to_layout(
                runtime_input_tensor, device, device_layout
            )
            runtime_host_tensor = ttrt.runtime.to_host(
                runtime_device_tensor, untilize=True, blocking=False
            )
            ttrt.runtime.wait(runtime_host_tensor)
            ttrt.runtime.deallocate_tensor(runtime_device_tensor, force=True)

            torch_output_tensor = torch.zeros_like(torch_input_tensor)
            ttrt.runtime.memcpy(
                torch_output_tensor.data_ptr(),
                runtime_input_tensor,
                Binary.Program.to_data_type(dtype),
            )

            assert torch.allclose(torch_input_tensor, torch_output_tensor)
