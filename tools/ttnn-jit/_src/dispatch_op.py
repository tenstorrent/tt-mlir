# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn_jit.runtime._ttmlir_runtime.runtime import (
    submit,
    set_compatible_runtime,
    get_current_runtime,
)
from ttnn_jit.runtime._ttmlir_runtime.binary import load_binary_from_path
from ttnn_jit.runtime._ttmlir_runtime.utils import (
    create_runtime_device_from_ttnn,
    create_runtime_tensor_from_ttnn,
    get_ttnn_tensor_from_runtime_tensor,
)


def _run_binary(binary_path, input_tensors):
    bin = load_binary_from_path(binary_path)
    set_compatible_device_runtime(bin)

    num_programs = bin.get_num_programs()
    if num_programs > 1:
        raise RuntimeError("Only one program is supported")

    device = input_tensors[0].device()
    runtime_device = create_runtime_device_from_ttnn(device)

    runtime_tensors = []
    for input_tensor in input_tensors:
        tensor = create_runtime_tensor_from_ttnn(input_tensor)
        runtime_tensors.append(tensor)

    program_name = bin.get_program_name(0)
    print(
        "Running program: ",
        program_name,
        " in runtime: ",
        get_current_device_runtime(),
    )

    output_runtime_tensor = submit(runtime_device, bin, 0, runtime_tensors)
    assert len(output_runtime_tensor) == 1, "Only one output tensor is supported"

    output_tensor = get_ttnn_tensor_from_runtime_tensor(output_runtime_tensor[0])
    return output_tensor
