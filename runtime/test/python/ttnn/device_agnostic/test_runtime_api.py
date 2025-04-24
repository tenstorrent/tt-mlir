# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import subprocess
from functools import partial
import ttrt
import ttrt.runtime
import ttrt.binary
import torch
from ttrt import API as ttrt_api
from ttrt.common.util import *
from ttnn.utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    assert_pcc,
    get_torch_inputs,
    get_runtime_tensor_from_torch,
    get_torch_output_container,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Silicon/TTNN/n150/runtime_stitching/Output"
)


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
    print("YOU ARE IN THE PRE OP CALLBACK")
    logging = callback_runtime_config.logging
    logging.debug("executing pre-op callback")
    op_intermediate_tensor_ids = ttrt.runtime.get_intermediate_input_tensor_ids(
        op_context
    )
    for tensor_id in op_intermediate_tensor_ids:  # restructure
        logging.debug(f"Intermediate input tensor id: {int(tensor_id)}")
        if ttrt.runtime.is_tensor_live(program_context, tensor_id):
            op_intermediate_tensors = ttrt.runtime.get_intermediate_input_tensors(
                op_context, program_context
            )
            op_intermediate_tensor_get = ttrt.runtime.get_tensor(
                program_context, tensor_id
            )
            logging.debug(f"Intermediate input tensors: {op_intermediate_tensors}")
            logging.debug(
                f"Intermediate input tensor method 2: {op_intermediate_tensor_get}"
            )
            # assert tensor_id in
            # Do I need to implement getTensorId from tensor?

            # for some reason, the tensors returned from the same ID have different object at values
            # assert op_intermediate_tensor_get in op_intermediate_tensors, f"Intermediate input tensors do not match. 1: {op_intermediate_tensors}, 2: {op_intermediate_tensor_get}"
        else:
            logging.debug("Input tensor is empty - skipping")

    input_tensor_ids = ttrt.runtime.get_input_tensor_ids(program_context)
    logging.debug(f"Input tensor ids: {input_tensor_ids}")
    output_tensor_ids = ttrt.runtime.get_output_tensor_ids(program_context)
    logging.debug(f"Output tensor ids: {output_tensor_ids}")
    logging.debug("Finished pre-op callback")


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_callback(callback_runtime_config, binary, program_context, op_context):
    print("YOU ARE IN THE POST OP CALLBACK")
    logging = callback_runtime_config.logging
    logging.debug("executing post-op callback")
    op_intermediate_tensor_ids = ttrt.runtime.get_intermediate_output_tensor_ids(
        op_context
    )
    for tensor_id in op_intermediate_tensor_ids:
        logging.debug(f"Intermediate output tensor id: {int(tensor_id)}")
        if ttrt.runtime.is_tensor_live(program_context, tensor_id):
            op_intermediate_tensor = ttrt.runtime.get_intermediate_output_tensor(
                op_context, program_context
            )
            op_intermediate_tensor_get = ttrt.runtime.get_tensor(
                program_context, tensor_id
            )
            logging.debug(f"Intermediate output tensor: {op_intermediate_tensor}")
            logging.debug(
                f"Intermediate output tensor method 2: {op_intermediate_tensor_get}"
            )
            assert (
                op_intermediate_tensor == op_intermediate_tensor_get
            ), f"Intermediate output tensors do not match. 1: {op_intermediate_tensor}, 2: {op_intermediate_tensor_get}"
        else:
            logging.debug("Output tensor is empty - skipping")
    logging.debug("Finished post-op callback")


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)


# to test: getOutputTensors(program context), get_intermediate_output_tensor, get_input/output_tensor_ids,
# get_intermediate_input_tensor_ids, is_tensor_live, get_tensor
# I think I still need to write get_intermediate_input_tensor in module.cpp

# in callback: getInputTensors(eventually)(only works before first op is run),
# get_intermediate_input_tensor_ids, get_intermediate_input_tensors(generally): pre callback only
# get_intermediate_output_tensor_ids, get_intermediate_output_tensors: post callback only (I think?)
# outside callback: getOutputTensos: run after program execution
# doesn't matter: getIn/OutputTensorIds, is_tensor_live, get_tensor

# I'd like to set this up such that it's scalable for future api runtime tests

# it would be nice to be able to run callback functions without having to hard code them in callback.py
# could I arrange for it to be passed in as a flag?

# see explorer runner.py for more info
def test_program_ttrt_apis(helper: Helper, request):
    # Add callback functions in here just in case
    import sys

    sys.path.append("/home/jgrim/wh-01-src/tt-mlir/runtime/test/python")
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "eltwise_binary_op_chain.mlir.tmp.ttnn"
    )
    binary_path = "ttnn/test_abs.ttnn"
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    # add save intermediate tensors flag
    ttrt_run_command = [
        "ttrt",
        "run",
        binary_path,
        "--log-file",
        "ttrt.log",
        "--import-callback-file",
        "ttnn.device_agnostic.test_runtime_api",
        "--import-pre-callback-function",
        "pre_op_get_callback_fn",
        "--import-post-callback-function",
        "post_op_get_callback_fn",
    ]

    print(f"Running command: {' '.join(ttrt_run_command)}")
    process = subprocess.Popen(
        ttrt_run_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    process.stdout.close()
    process.wait()
    assert False, "test"


# This is the python path that works for me: /home/jgrim/wh-01-src/tt-mlir/runtime/test/python:/home/jgrim/wh-01-src/tt-mlir/build/python_packages:/home/jgrim/wh-01-src/tt-mlir/.local/toolchain/python_packages/mlir_core:/home/jgrim/wh-01-src/tt-mlir/third_party/tt-metal/src/tt-metal:/home/jgrim/wh-01-src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_eager:/home/jgrim/wh-01-src/tt-mlir/third_party/tt-metal/src/tt-metal-build/tools/profiler/bin:/home/jgrim/wh-01-src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn