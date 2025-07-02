# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import threading
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from .constants import FLATBUFFER_BASE_PATH

from ..utils import (
    Helper,
    DeviceContext,
    assert_pcc,
    get_torch_inputs,
    get_runtime_tensor_from_torch,
    get_torch_output_container,
    get_to_layout_inputs,
)


def worker_fn(binary, torch_inputs, mesh_device, results):
    program = binary.get_program(0)
    runtime_inputs = [
        get_runtime_tensor_from_torch(torch_input) for torch_input in torch_inputs
    ]
    runtime_inputs = get_to_layout_inputs(mesh_device, runtime_inputs, binary, 0)
    torch_output = get_torch_output_container(program)
    runtime_output = ttrt.runtime.submit(mesh_device, binary.fbb, 0, runtime_inputs)
    output_host = ttrt.runtime.to_host(runtime_output[0], untilize=True)[0]
    ttrt.runtime.deallocate_tensor(runtime_output[0], force=True)
    ttrt.runtime.memcpy(torch_output.data_ptr(), output_host)
    ttrt.runtime.deallocate_tensor(output_host, force=True)
    results.append(torch_output)


def test_eltwise_binary_add_data_parallel(helper: Helper, request):
    num_devices = ttrt.runtime.get_num_available_devices()
    assert num_devices == 2, "Test requires 2 devices"
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program: Binary.Program = helper.binary.get_program(0)
    assert program.num_inputs() == 2
    inputs_torch = [get_torch_inputs(program) for _ in range(2)]
    batched_tensors = [
        torch.stack([t1, t2], dim=0) for t1, t2 in zip(inputs_torch[0], inputs_torch[1])
    ]
    assert len(batched_tensors) == 2

    with DeviceContext(mesh_shape=[1, 2]) as parent_mesh:
        threads = []
        results = []
        submeshes = []
        for i in range(2):
            inputs = [batched_tensors[0][i], batched_tensors[1][i]]
            results.append([])
            submeshes.append(
                ttrt.runtime.create_sub_mesh_device(
                    parent_mesh, mesh_shape=[1, 1], mesh_offset=(0, i)
                )
            )
            thread = threading.Thread(
                target=worker_fn, args=(helper.binary, inputs, submeshes[i], results[i])
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        for submesh in submeshes:
            ttrt.runtime.release_sub_mesh_device(submesh)

        submeshes.clear()

    batched_results = torch.stack([result[0] for result in results], dim=0)
    batched_golden = torch.add(batched_tensors[0], batched_tensors[1])
    assert_pcc(batched_golden, batched_results, threshold=0.99)
    helper.teardown()
