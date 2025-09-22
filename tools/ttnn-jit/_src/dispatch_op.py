# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from _ttmlir_runtime import runtime, binary, utils


def _run_binary(binary_path, input_tensors):
    bin = binary.load_binary_from_path(binary_path)
    runtime.set_compatible_runtime(bin)

    num_programs = bin.get_num_programs()
    if num_programs > 1:
        raise RuntimeError("Only one program is supported")

    device = input_tensors[0].device()
    runtime_device = utils.create_runtime_device_from_ttnn(device)

    runtime_tensors = []
    for input_tensor in input_tensors:
        tensor = utils.create_runtime_tensor_from_ttnn(input_tensor)
        runtime_tensors.append(tensor)

    program_name = bin.get_program_name(0)
    print(
        "Running program: ",
        program_name,
        " in runtime: ",
        runtime.get_current_runtime(),
    )

    output_runtime_tensor = runtime.submit(runtime_device, bin, 0, runtime_tensors)
    output_tensor = utils.get_ttnn_tensor_from_runtime_tensor(output_runtime_tensor[0])
    return output_tensor
