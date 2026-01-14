# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: ttnn-jit

import ttnn_jit
import ttnn
import torch

from utils import create_sharded_tile_tensor, create_dram_tensor


@ttnn_jit.jit(compile_only=True)
def abs(input_tensor):
    return ttnn.abs(input_tensor)


@ttnn_jit.jit(compile_only=True)
def add(input_tensor_a, input_tensor_b):
    return ttnn.add(input_tensor_a, input_tensor_b)


@ttnn_jit.jit(compile_only=True)
def reduce_max(input_tensor):
    return ttnn.max(input_tensor, dim=1, keepdim=True)


@ttnn_jit.jit(compile_only=True)
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

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: #ttnn.buffer_type<l1>
    # CHECK: func.func @abs
    # CHECK: ttir.abs
    _ = abs(input_tensor_a_l1)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: #ttnn.buffer_type<dram>
    # CHECK: #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>{{.*}} <interleaved>
    # CHECK: func.func @abs
    # CHECK: ttir.abs
    _ = abs(input_tensor_a_dram)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @add
    # CHECK: ttir.add
    _ = add(input_tensor_a_l1, input_tensor_b_l1)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @reduce_max{{.*}} -> tensor<128x1xbf16, {{.*}}>
    # CHECK: ttir.max
    _ = reduce_max(input_tensor_a_l1)

    # CHECK: ---- IR Dump after TracingCompiler (Tracing-based) ----
    # CHECK: func.func @matmul{{.*}} -> tensor<128x256xbf16, {{.*}}>
    # CHECK: ttir.matmul
    _ = matmul(input_tensor_a_l1, input_tensor_c_l1)

    ttnn.close_device(device)
