// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=32,4" --ttnn-fusing="enable-ring-sdpa=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// TP=4 and SP=32 is the only mesh that should select the experimental kernel.

#dram = #ttnn.buffer_type<dram>
#q = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1280 + d1 * 128 + d2, d3), <1x1>, memref<40x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_bshd = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1280 + d1 * 10 + d2, d3), <1x1>, memref<40x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#gathered = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 40960 + d1 * 10 + d2, d3), <1x1>, memref<1280x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#trimmed = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 40000 + d1 * 10 + d2, d3), <1x1>, memref<1250x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_bhnd = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 40000 + d1 * 4000 + d2, d3), <1x1>, memref<1250x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @sp32_tp4_uses_exp_ring_joint(
      %q: tensor<1x10x128x64xbf16, #q>,
      %k: tensor<1x128x10x64xbf16, #kv_bshd>,
      %v: tensor<1x128x10x64xbf16, #kv_bshd>)
      -> tensor<1x10x128x64xbf16, #q> {
    // CHECK-LABEL: @sp32_tp4_uses_exp_ring_joint
    // CHECK-NOT: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.ring_joint_scaled_dot_product_attention"
    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"(%arg0,
    // CHECK-SAME: cluster_axis = 0 : ui32
    // CHECK-SAME: logical_n = 4000 : i64
    // Exp kernel uses the full compute grid; it does not take a CCL offset.
    // CHECK-SAME: compute_with_storage_grid_size = <8, 8>
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}> : (tensor<1x128x10x64xbf16, #kv_bshd>) -> tensor<1x4096x10x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}> : (tensor<1x128x10x64xbf16, #kv_bshd>) -> tensor<1x4096x10x64xbf16, #gathered>
    %2 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4000 : i32, 10 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4096x10x64xbf16, #gathered>) -> tensor<1x4000x10x64xbf16, #trimmed>
    %3 = "ttnn.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4000 : i32, 10 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4096x10x64xbf16, #gathered>) -> tensor<1x4000x10x64xbf16, #trimmed>
    %4 = "ttnn.permute"(%2) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4000x10x64xbf16, #trimmed>) -> tensor<1x10x4000x64xbf16, #kv_bhnd>
    %5 = "ttnn.permute"(%3) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4000x10x64xbf16, #trimmed>) -> tensor<1x10x4000x64xbf16, #kv_bhnd>
    %6 = "ttnn.scaled_dot_product_attention"(%q, %4, %5) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x10x128x64xbf16, #q>, tensor<1x10x4000x64xbf16, #kv_bhnd>, tensor<1x10x4000x64xbf16, #kv_bhnd>) -> tensor<1x10x128x64xbf16, #q>
    return %6 : tensor<1x10x128x64xbf16, #q>
  }
}
