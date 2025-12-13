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


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=True)
def reduce_max(input_tensor):
    return ttnn.max(input_tensor, dim=1, keepdim=True)


@ttnn_jit.jit(debug=True, compile_only=True, graph_capture=False)
def matmul(input_tensor_a, input_tensor_b):
    return ttnn.matmul(input_tensor_a, input_tensor_b)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    input_tensor_a_l1 = create_sharded_tile_tensor(
        device, (128, 128), (0, 0), torch.bfloat16
    )
    input_tensor_b_l1 = create_sharded_tile_tensor(
        device, (128, 128), (0, 0), torch.bfloat16
    )
    input_tensor_c_l1 = create_sharded_tile_tensor(
        device, (128, 256), (0, 0), torch.bfloat16
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

    # CHECK:---- IR Dump after GraphToIRTranslator (Graph-based) ----
    # CHECK: func.func @reduce_max{{.*}} -> tensor<128x1xbf16, {{.*}}>
    # CHECK: "ttnn.max"{{.*}} <{dim_arg = [1 : i32], keep_dim = true}> {ttnn.hoist_generic_via_d2m}
    _ = reduce_max(input_tensor_a_l1)

    # CHECK: ---- IR Dump after TTIRCompiler (AST-based) ----
    # CHECK: func.func @matmul{{.*}} -> tensor<128x256xbf16, {{.*}}>
    # CHECK: "ttnn.matmul"{{.*}} <{transpose_a = false, transpose_b = false}> {dtype = #ttcore.supportedDataTypes<bf16>, ttnn.hoist_generic_via_d2m}
    _ = matmul(input_tensor_a_l1, input_tensor_c_l1)

    ttnn.close_device(device)
