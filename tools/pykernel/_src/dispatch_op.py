# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from _ttmlir_runtime import runtime as runtime, binary as binary, utils as utils


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


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    shape = [1, 1, 32, 32]
    data0 = torch.randn(shape).to(torch.float32)

    input_tensor = ttnn.from_torch(
        data0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    """
    cos.ttnn is hand-generated as follows:
    ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYSTEM_DESC_PATH}" cos.mlir 2>&1 | tee cos_lowered.mlir
    ttmlir-translate --ttnn-to-flatbuffer cos_lowered.mlir > cos.ttnn
    func.func @cosine(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %0 = ttir.empty() : tensor<32x32xf32>
        %1 = "ttir.cos"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
    }
    """
    output_tensor = _run_binary("cos.ttnn", [input_tensor])
    print(input_tensor)
    print(output_tensor)

    golden = torch.cos(data0.cpu())
    print(golden)

    all_close = torch.allclose(output_tensor.cpu().to_torch(), golden, atol=1e-4)
    print(all_close)

    ttnn.close_device(device)
