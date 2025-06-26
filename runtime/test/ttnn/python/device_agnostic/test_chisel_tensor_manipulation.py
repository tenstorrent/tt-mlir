# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttrt
import ttmlir
import torch
import os
import tempfile

from ttrt.common.api import API as RtApi
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts
from ttrt.runtime import (
    DebugHooks,
    get_op_debug_str,
    get_op_output_tensor,
    get_op_output_ref,
    get_op_input_refs,
    retrieve_tensor_from_pool,
    update_tensor_in_pool,
    memcpy,
    create_owned_host_tensor,
    DataType,
    unregister_hooks,
)
from ttir_builder import TTIRBuilder, Operand, GoldenCheckLevel
from ttir_builder.utils import compile_to_flatbuffer
from ..utils import TT_MLIR_HOME

# Constants
INPUT_SHAPE = (10, 10)
WEIGHT_SHAPE = (10, 10)
BIAS_SHAPE = (1, 10)
DTYPE = torch.float32


def get_torch_tensor(tensor):
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    if rt_dtype is not DataType.Float32:
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
    tensor = create_owned_host_tensor(data_ptr, shape, stride, size, dtype)
    update_tensor_in_pool(program_context, tensor_ref, tensor)


def linear_model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    builder._golden_check_level = GoldenCheckLevel.DISABLED
    matmul_result = builder.matmul(in0, in1)
    output = builder.add(matmul_result, in2)
    return output


@pytest.mark.parametrize(
    "golden_input_tensors,golden_out_tensors,update_in_tensors",
    [
        (
            {
                0: torch.ones(INPUT_SHAPE),
                1: torch.ones(WEIGHT_SHAPE),
                2: torch.ones(BIAS_SHAPE),
                3: torch.ones(WEIGHT_SHAPE),
                4: torch.ones(WEIGHT_SHAPE) * 10,
                5: torch.ones(BIAS_SHAPE),
                6: torch.ones(WEIGHT_SHAPE) * 20,
                7: torch.ones(BIAS_SHAPE),
            },
            {0: torch.ones(WEIGHT_SHAPE) * 10, 1: torch.ones(WEIGHT_SHAPE) * 21},
            {4: torch.ones(WEIGHT_SHAPE) * 20},
        ),
    ],
)
def test_tensor_manipulation_apis(
    golden_input_tensors, golden_out_tensors, update_in_tensors
):
    output_root = "."
    test_base = "test_chisel_tensor_manipulation"
    compile_to_flatbuffer(
        linear_model,
        [INPUT_SHAPE, WEIGHT_SHAPE, BIAS_SHAPE],
        target="ttnn",
        output_root=output_root,
        test_base=test_base,
        golden_check_level=GoldenCheckLevel.DISABLED,
    )
    flatbuffer_path = f"{output_root}/ttnn/{test_base}_ttnn.mlir.ttnn"
    assert os.path.exists(flatbuffer_path)

    in_counter = 0
    out_counter = 0

    def preop(binary, programContext, opContext):
        nonlocal in_counter
        tensor_refs = get_op_input_refs(opContext, programContext)

        for ref in tensor_refs:
            tensor = retrieve_tensor_from_pool(programContext, ref)
            torch_tensor = get_torch_tensor(tensor)
            if in_counter in golden_input_tensors:
                print(f"Input {in_counter}: {torch_tensor}")
                print(f"Golden {in_counter}: {golden_input_tensors[in_counter]}")
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
        if out_counter in golden_out_tensors:
            print(f"Output {out_counter}: {torch_tensor}")
            print(f"Golden {out_counter}: {golden_out_tensors[out_counter]}")
            assert torch.all(torch_tensor == golden_out_tensors[out_counter])
        out_counter += 1

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
