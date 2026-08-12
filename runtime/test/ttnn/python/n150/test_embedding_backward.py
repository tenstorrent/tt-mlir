# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttrt
import ttrt.runtime
from ttrt.common.util import *
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    assert_pcc,
    get_runtime_tensor_from_torch,
    get_to_layout_inputs,
    get_torch_output_container,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/embedding_backward/Output"
)

VOCAB_SIZE = 512
EMBEDDING_DIM = 128
SEQ_LEN = 32


def test_embedding_backward_output_matches_declared_shape(helper: Helper, request):
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "embedding_backward.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program: Binary.Program = helper.binary.get_program(0)
    assert program.num_inputs() == 3 and program.num_outputs() == 1

    torch.manual_seed(0)
    indices = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), dtype=torch.int32)
    weight = torch.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=torch.float32)
    in_gradient = torch.randn((1, SEQ_LEN, EMBEDDING_DIM), dtype=torch.float32)
    inputs_torch = [indices, weight, in_gradient]

    golden = torch.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=torch.float32)
    golden.index_add_(
        0, indices.flatten().long(), in_gradient.reshape(-1, EMBEDDING_DIM)
    )

    declared_shape = program.outputs[0]["desc"]["shape"]
    assert declared_shape == [VOCAB_SIZE, EMBEDDING_DIM]

    torch_result = get_torch_output_container(program)

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime = [
            get_runtime_tensor_from_torch(input) for input in inputs_torch
        ]
        inputs = get_to_layout_inputs(device, inputs_runtime, helper.binary, 0)
        outputs = ttrt.runtime.submit(device, helper.binary.fbb, 0, inputs)
        assert outputs[0].get_shape() == declared_shape
        host_output = ttrt.runtime.to_host(outputs[0], untilize=True)[0]
        ttrt.runtime.memcpy(torch_result.data_ptr(), host_output)
        ttrt.runtime.deallocate_tensor(outputs[0], force=True)
        ttrt.runtime.deallocate_tensor(host_output, force=True)

    assert_pcc(golden, torch_result, threshold=0.99)
    helper.teardown()
