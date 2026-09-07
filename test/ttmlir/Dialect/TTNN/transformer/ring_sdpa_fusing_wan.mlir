// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=8,4" --ttnn-fusing="enable-ring-sdpa=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Wan Graph A self-attn on Galaxy 8x4 (SP=8 on cluster_axis 0, TP=4):
// gather K/V in [B, S, H, D], trim pad, permute to [B, H, S, D], then SDPA.
// Must emit the non-experimental ring_joint kernel, not exp_ring_joint.

#dram = #ttnn.buffer_type<dram>
#q = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1280 + d1 * 128 + d2, d3), <1x1>, memref<40x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_bshd = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1280 + d1 * 10 + d2, d3), <1x1>, memref<40x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#gathered = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 10240 + d1 * 10 + d2, d3), <1x1>, memref<320x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#trimmed = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 10000 + d1 * 10 + d2, d3), <1x1>, memref<313x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_bhnd = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 10000 + d1 * 1000 + d2, d3), <1x1>, memref<313x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @wan_sp8_uses_ring_joint(
      %q: tensor<1x10x128x64xbf16, #q>,
      %k: tensor<1x128x10x64xbf16, #kv_bshd>,
      %v: tensor<1x128x10x64xbf16, #kv_bshd>)
      -> tensor<1x10x128x64xbf16, #q> {
    // CHECK-LABEL: @wan_sp8_uses_ring_joint
    // CHECK-NOT: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK-NOT: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // CHECK: "ttnn.ring_joint_scaled_dot_product_attention"(%arg0,
    // CHECK-SAME: cluster_axis = 0 : ui32
    // CHECK-SAME: dim = 2 : si32
    // CHECK-SAME: logical_n = 1000 : i64
    // CHECK-SAME: compute_with_storage_grid_size = <7, 8>
    // CHECK-SAME: q_chunk_size = 128
    // CHECK-SAME: k_chunk_size = 256
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}> : (tensor<1x128x10x64xbf16, #kv_bshd>) -> tensor<1x1024x10x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}> : (tensor<1x128x10x64xbf16, #kv_bshd>) -> tensor<1x1024x10x64xbf16, #gathered>
    %2 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1000 : i32, 10 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x10x64xbf16, #gathered>) -> tensor<1x1000x10x64xbf16, #trimmed>
    %3 = "ttnn.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1000 : i32, 10 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x10x64xbf16, #gathered>) -> tensor<1x1000x10x64xbf16, #trimmed>
    %4 = "ttnn.permute"(%2) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1000x10x64xbf16, #trimmed>) -> tensor<1x10x1000x64xbf16, #kv_bhnd>
    %5 = "ttnn.permute"(%3) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1000x10x64xbf16, #trimmed>) -> tensor<1x10x1000x64xbf16, #kv_bhnd>
    %6 = "ttnn.scaled_dot_product_attention"(%q, %4, %5) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x10x128x64xbf16, #q>, tensor<1x10x1000x64xbf16, #kv_bhnd>, tensor<1x10x1000x64xbf16, #kv_bhnd>) -> tensor<1x10x128x64xbf16, #q>
    return %6 : tensor<1x10x128x64xbf16, #q>
  }

  func.func @wan_skips_to_layout(
      %q: tensor<1x10x128x64xbf16, #q>,
      %k: tensor<1x128x10x64xbf16, #kv_bshd>,
      %v: tensor<1x128x10x64xbf16, #kv_bshd>)
      -> tensor<1x10x128x64xbf16, #q> {
    // CHECK-LABEL: @wan_skips_to_layout
    // CHECK: "ttnn.ring_joint_scaled_dot_product_attention"
    // CHECK-SAME: logical_n = 1000 : i64
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}> : (tensor<1x128x10x64xbf16, #kv_bshd>) -> tensor<1x1024x10x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}> : (tensor<1x128x10x64xbf16, #kv_bshd>) -> tensor<1x1024x10x64xbf16, #gathered>
    %2 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1000 : i32, 10 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x10x64xbf16, #gathered>) -> tensor<1x1000x10x64xbf16, #trimmed>
    %3 = "ttnn.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1000 : i32, 10 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x10x64xbf16, #gathered>) -> tensor<1x1000x10x64xbf16, #trimmed>
    %4 = "ttnn.to_layout"(%2) : (tensor<1x1000x10x64xbf16, #trimmed>) -> tensor<1x1000x10x64xbf16, #trimmed>
    %5 = "ttnn.to_layout"(%3) : (tensor<1x1000x10x64xbf16, #trimmed>) -> tensor<1x1000x10x64xbf16, #trimmed>
    %6 = "ttnn.permute"(%4) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1000x10x64xbf16, #trimmed>) -> tensor<1x10x1000x64xbf16, #kv_bhnd>
    %7 = "ttnn.permute"(%5) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1000x10x64xbf16, #trimmed>) -> tensor<1x10x1000x64xbf16, #kv_bhnd>
    %8 = "ttnn.to_layout"(%6) : (tensor<1x10x1000x64xbf16, #kv_bhnd>) -> tensor<1x10x1000x64xbf16, #kv_bhnd>
    %9 = "ttnn.to_layout"(%7) : (tensor<1x10x1000x64xbf16, #kv_bhnd>) -> tensor<1x10x1000x64xbf16, #kv_bhnd>
    %10 = "ttnn.scaled_dot_product_attention"(%q, %8, %9) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x10x128x64xbf16, #q>, tensor<1x10x1000x64xbf16, #kv_bhnd>, tensor<1x10x1000x64xbf16, #kv_bhnd>) -> tensor<1x10x128x64xbf16, #q>
    return %10 : tensor<1x10x128x64xbf16, #q>
  }
}
