# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: ttnn-jit

import ttnn_jit
import ttnn
import torch

from utils import create_sharded_tile_tensor, create_dram_tensor


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=False)
def abs(input_tensor):
    return ttnn.abs(input_tensor)


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=False)
def add(input_tensor_a, input_tensor_b):
    return ttnn.add(input_tensor_a, input_tensor_b)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    input_tensor_a_l1 = create_sharded_tile_tensor(
        device, (128, 128), (0, 0), torch.bfloat16
    )
    input_tensor_b_l1 = create_sharded_tile_tensor(
        device, (128, 128), (0, 0), torch.bfloat16
    )
    input_tensor_a_dram = create_dram_tensor(device, (128, 128), torch.bfloat16)
    input_tensor_b_dram = create_dram_tensor(device, (128, 128), torch.bfloat16)

    # CHECK: ---- IR Dump after TTIRCompiler (AST-based) ----
    # CHECK: #ttnn.buffer_type<l1>
    # CHECK: func.func @abs
    # CHECK: "ttnn.abs"(%arg0) {ttnn.hoist_generic_via_d2m}
    _ = abs(input_tensor_a_l1)

    # CHECK: ---- IR Dump after TTIRCompiler (AST-based) ----
    # CHECK: #ttnn.buffer_type<dram>
    # CHECK: #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>{{.*}} <interleaved>
    # CHECK: func.func @abs
    # CHECK: "ttnn.abs"(%arg0) {ttnn.hoist_generic_via_d2m}
    _ = abs(input_tensor_a_dram)

    # CHECK: ---- IR Dump after TTIRCompiler (AST-based) ----
    # CHECK: func.func @add
    # CHECK: "ttnn.add"{{.*}} <{dtype = #ttcore.supportedDataTypes<bf16>}> {ttnn.hoist_generic_via_d2m}
    _ = add(input_tensor_a_l1, input_tensor_b_l1)

    ttnn.close_device(device)
