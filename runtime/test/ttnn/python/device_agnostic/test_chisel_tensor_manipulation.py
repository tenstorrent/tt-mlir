# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttrt
import ttmlir
import torch
import pdb
import os

from ttrt.common.api import API as RtApi
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts
from ttrt.runtime import (
    DebugHooks,
    get_op_debug_str,
    get_op_output_tensor,
    get_op_output_ref,
    get_op_input_refs,
    get_tensor,
    update_tensor,
    memcpy,
    create_tensor,
    DataType,
    unregister_hooks,
)
from ..utils import TT_MLIR_HOME

DIRECTORY_PATH = f"{TT_MLIR_HOME}/build/test/ttmlir/Silicon"

DTYPE_TO_TORCH_DTYPE = {
    DataType.Float32: torch.float32,
    DataType.Float16: torch.float16,
    DataType.BFloat16: torch.bfloat16,
    # DataType.BFP_Float8
    # DataType.BFP_BFloat8
    # DataType.BFP_Float4
    # DataType.BFP_BFloat4
    # DataType.BFP_Float2
    # DataType.BFP_BFloat2
    DataType.UInt32: torch.uint32,
    DataType.UInt16: torch.uint16,
    DataType.UInt8: torch.uint8,
    DataType.Int32: torch.int32,
}


def get_torch_tensor(tensor):
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = DTYPE_TO_TORCH_DTYPE[rt_dtype]
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
    tensor = create_tensor(data_ptr, shape, stride, size, dtype)
    update_tensor(program_context, tensor_ref, tensor)


@pytest.mark.parametrize(
    "flatbuffer_path,golden_input_tensors,golden_out_tensors,update_in_tensors",
    [
        (
            f"{DIRECTORY_PATH}/TTNN/n150/chisel/Output/test_tensor_manipulation.mlir.tmp.ttnn",
            {
                0: torch.ones([10, 10]),
                1: torch.ones([10, 10]),
                2: torch.ones([10, 10]),
                3: torch.ones([10, 10]),
                4: torch.ones([10, 10]) * 10,
                5: torch.ones([10]),
                6: torch.ones([10, 10]) * 20,
                7: torch.ones([10]),
            },
            {0: torch.ones([10, 10]) * 10, 1: torch.ones([10, 10]) * 21},
            {4: torch.ones([10, 10]) * 20},
        ),
    ],
)
def test_tensor_manipulation_apis(
    flatbuffer_path, golden_input_tensors, golden_out_tensors, update_in_tensors
):
    in_counter = 0
    out_counter = 0

    def preop(binary, programContext, opContext):
        nonlocal in_counter
        tensor_refs = get_op_input_refs(opContext, programContext)

        for ref in tensor_refs:
            tensor = get_tensor(programContext, ref)
            torch_tensor = get_torch_tensor(tensor)
            if in_counter in golden_input_tensors:
                assert torch.all(torch_tensor == golden_input_tensors[in_counter])
            if in_counter in update_in_tensors:
                update_device_tensor(
                    programContext, ref, tensor, update_in_tensors[in_counter]
                )
            in_counter += 1

    def postop(binary, programContext, opContext):
        nonlocal out_counter

        tensor = get_op_output_tensor(opContext, programContext)
        if tensor is None:
            return
        torch_tensor = get_torch_tensor(tensor)
        print(torch_tensor)
        if out_counter in golden_out_tensors:
            assert torch.all(torch_tensor == golden_out_tensors[out_counter])
        out_counter += 1

    assert os.path.exists(flatbuffer_path)

    args = {
        "binary": str(flatbuffer_path),
        "save-artifacts": True,
        "--init": "ones",
    }
    rt_logger = RtLogger()
    rt_artifacts = RtArtifacts(logger=rt_logger)
    RtApi.initialize_apis()
    rt_api = RtApi.Run(args=args, logger=rt_logger, artifacts=rt_artifacts)

    callback_env_pre = DebugHooks.get(preop, postop)
    result_code, results = rt_api()
    unregister_hooks()
    assert result_code == 0, "Test failed"
