// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file %s | FileCheck %s

module attributes {} {
  func.func @ring_joint_self_attention(
      %q: tensor<1x40x512x128xbf16>,
      %k: tensor<1x40x512x128xbf16>,
      %v: tensor<1x40x512x128xbf16>,
      %buf_k: tensor<1x40x1024x128xbf16>,
      %buf_v: tensor<1x40x1024x128xbf16>,
      %ping: !ttnn.global_semaphore,
      %pong: !ttnn.global_semaphore)
      -> tensor<1x40x512x128xbf16> {
    // CHECK-LABEL: @ring_joint_self_attention
    // CHECK: "ttnn.ring_joint_scaled_dot_product_attention"
    // CHECK-SAME: cluster_axis = 0 : ui32
    // CHECK-SAME: dim = 2 : si32
    // CHECK-SAME: joint_strategy = "rear"
    // CHECK-SAME: logical_n = 1000 : i64
    %0, %1, %2 = "ttnn.ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping, %pong) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>,
      joint_strategy = "rear",
      logical_n = 1000 : i64,
      dim = 2 : si32,
      cluster_axis = 0 : ui32,
      program_config = #ttnn.sdpa_program_config<
        compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
        q_chunk_size = 128,
        k_chunk_size = 128>
    }> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x1024x128xbf16>, tensor<1x40x1024x128xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x0x128xbf16>, tensor<1x40x1024x1xbf16>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
