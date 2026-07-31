// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=1,2" --ttnn-fusing="enable-ring-sdpa=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Phase 4: absorbing the frontend's padding trim into logical_n.
//
// A frontend that pads the sequence to TILE_SIZE * SP has to slice the gathered
// K/V back to the true length before every block, because plain SDPA cannot
// tell padded keys from real ones. The ring op takes logical_n instead, so the
// slice becomes the carrier of the true length and disappears.
//
// Sequence is padded to 256 (2 devices x 128) with 200 real tokens.
//
// NOTE: a K-only trim is deliberately not tested. The plain SDPA verifier
// requires key and value to have the same shape, so slicing one and not the
// other cannot be expressed in valid IR -- the guard for it is defensive only,
// like the is_causal guard in ring_sdpa_fusing_negative.mlir.

#dram = #ttnn.buffer_type<dram>
#sharded = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#gathered = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 256 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#trimmed = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1600 + d1 * 200 + d2, d3), <1x1>, memref<50x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#short = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 96 + d2, d3), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#halfgathered = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 256 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // The slice is absorbed: logical_n carries 200 and both slice_static ops go.
  func.func @absorbs_padding_slice(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x128x64xbf16, #sharded> {
    // CHECK-LABEL: @absorbs_padding_slice
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK-NOT: "ttnn.all_gather"
    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"(%arg0, %arg1, %arg2)
    // CHECK-SAME: logical_n = 200 : i64
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 200 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x200x64xbf16, #trimmed>
    %3 = "ttnn.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 200 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x200x64xbf16, #trimmed>
    %4 = "ttnn.scaled_dot_product_attention"(%q, %2, %3) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x200x64xbf16, #trimmed>, tensor<1x8x200x64xbf16, #trimmed>) -> tensor<1x8x128x64xbf16, #sharded>
    return %4 : tensor<1x8x128x64xbf16, #sharded>
  }

  // No slice at all: logical_n is the whole gathered length.
  func.func @no_slice_means_no_padding(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x128x64xbf16, #sharded> {
    // CHECK-LABEL: @no_slice_means_no_padding
    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // CHECK-SAME: logical_n = 256 : i64
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>, tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x128x64xbf16, #sharded>
    return %2 : tensor<1x8x128x64xbf16, #sharded>
  }

  // A trim shorter than one shard would leave device 1 holding only padding.
  // tt-metal rejects that (TT_FATAL((N_global - logical_n) < N_local)), so the
  // rewrite declines rather than emitting an op that cannot run.
  func.func @trim_below_one_shard(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x128x64xbf16, #sharded> {
    // CHECK-LABEL: @trim_below_one_shard
    // CHECK-NOT: exp_ring_joint
    // CHECK: "ttnn.scaled_dot_product_attention"
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 96 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x96x64xbf16, #short>
    %3 = "ttnn.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 96 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x96x64xbf16, #short>
    %4 = "ttnn.scaled_dot_product_attention"(%q, %2, %3) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x96x64xbf16, #short>, tensor<1x8x96x64xbf16, #short>) -> tensor<1x8x128x64xbf16, #sharded>
    return %4 : tensor<1x8x128x64xbf16, #sharded>
  }

  // A slice that also trims heads is not a padding trim, so it is not absorbed
  // and the plain form stands.
  func.func @slice_trims_heads(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x128x64xbf16, #sharded> {
    // CHECK-LABEL: @slice_trims_heads
    // CHECK-NOT: exp_ring_joint
    // CHECK: "ttnn.scaled_dot_product_attention"
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4 : i32, 256 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x4x256x64xbf16, #halfgathered>
    %3 = "ttnn.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4 : i32, 256 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x4x256x64xbf16, #halfgathered>
    %4 = "ttnn.scaled_dot_product_attention"(%q, %2, %3) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x4x256x64xbf16, #halfgathered>, tensor<1x4x256x64xbf16, #halfgathered>) -> tensor<1x8x128x64xbf16, #sharded>
    return %4 : tensor<1x8x128x64xbf16, #sharded>
  }
}
