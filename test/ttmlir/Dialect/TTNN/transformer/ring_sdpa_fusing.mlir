// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=1,2" --ttnn-fusing="enable-ring-sdpa=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// The rewrite that turns an exposed sequence-parallel K/V all-gather feeding a
// plain SDPA into the ring op. Q stays sequence-sharded at [1,8,128,64]; the
// all-gathers take K/V to the full [1,8,256,64] on a 2-device ring.

#dram = #ttnn.buffer_type<dram>

#sharded_layout = #ttnn.ttnn_layout<
  (d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3),
  <1x1>,
  memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>
>

#gathered_layout = #ttnn.ttnn_layout<
  (d0, d1, d2, d3) -> (d0 * 2048 + d1 * 256 + d2, d3),
  <1x1>,
  memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>
>

module {
  func.func @ring_sdpa_fires(
      %q: tensor<1x8x128x64xbf16, #sharded_layout>,
      %k: tensor<1x8x128x64xbf16, #sharded_layout>,
      %v: tensor<1x8x128x64xbf16, #sharded_layout>)
      -> tensor<1x8x128x64xbf16, #sharded_layout> {
    // CHECK-LABEL: @ring_sdpa_fires
    // Both all-gathers and the plain SDPA are gone.
    // CHECK-NOT: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.scaled_dot_product_attention"
    //
    // The ring op takes the PRE-gather K/V, and the CCL attributes are lifted
    // straight off the matched all-gathers.
    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"(%arg0, %arg1, %arg2)
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: dim = 2 : si32
    // CHECK-SAME: joint_strategy = "rear"
    // No padding slice absorbed yet, so logical_n is the whole gathered length.
    // CHECK-SAME: logical_n = 256 : i64
    // CHECK-SAME: num_buffers_per_channel = 32 : ui32
    // CHECK-SAME: num_links = 1 : ui32
    // CHECK-SAME: num_workers_per_link = 5 : ui32
    // Buffers and semaphores are left unbound for the prelude passes.
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0>
    // CHECK-SAME: q_chunk_size = 128
    // CHECK-SAME: k_chunk_size = 256
    // CHECK-SAME: topology = #ttcore.topology<ring>
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32, num_links = 1 : ui32, topology = #ttcore.topology<ring>}> : (tensor<1x8x128x64xbf16, #sharded_layout>) -> tensor<1x8x256x64xbf16, #gathered_layout>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32, num_links = 1 : ui32, topology = #ttcore.topology<ring>}> : (tensor<1x8x128x64xbf16, #sharded_layout>) -> tensor<1x8x256x64xbf16, #gathered_layout>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded_layout>, tensor<1x8x256x64xbf16, #gathered_layout>, tensor<1x8x256x64xbf16, #gathered_layout>) -> tensor<1x8x128x64xbf16, #sharded_layout>
    return %2 : tensor<1x8x128x64xbf16, #sharded_layout>
  }
}
