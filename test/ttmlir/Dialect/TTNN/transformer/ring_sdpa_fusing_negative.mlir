// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=1,2" --ttnn-fusing="enable-ring-sdpa=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
//
// And with the option left at its default, nothing may fire anywhere.
// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=1,2" --ttnn-fusing -o %t_off.mlir %s
// RUN: FileCheck %s --check-prefix=OFF --input-file=%t_off.mlir

// Cases the ring rewrite must decline. Each must leave the plain
// all_gather + scaled_dot_product_attention form untouched.

// OFF-NOT: exp_ring_joint

#dram = #ttnn.buffer_type<dram>
#sharded = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#gathered = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 256 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#heads = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 128 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#mask = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// NOTE: there is deliberately no is_causal case here. `is_causal` requires
// Sq == Sk, but a gathered K/V is longer than the sequence-sharded Q, so the
// SDPA verifier rejects that combination before the pattern can see it. The
// pattern's is_causal guard is therefore defensive only and cannot be
// exercised through valid IR.

module {
  // An explicit attention mask is not supported.
  func.func @no_fire_mask(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>, %m: tensor<1x1x128x256xbf16, #mask>) -> tensor<1x8x128x64xbf16, #sharded> {
    // CHECK-LABEL: @no_fire_mask
    // CHECK-NOT: exp_ring_joint
    // CHECK: "ttnn.scaled_dot_product_attention"
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1, %m) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>, tensor<1x8x256x64xbf16, #gathered>, tensor<1x1x128x256xbf16, #mask>) -> tensor<1x8x128x64xbf16, #sharded>
    return %2 : tensor<1x8x128x64xbf16, #sharded>
  }

  // The two all-gathers disagree on cluster_axis, so they are not one ring.
  func.func @no_fire_axis_mismatch(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x128x64xbf16, #sharded> {
    // CHECK-LABEL: @no_fire_axis_mismatch
    // CHECK-NOT: exp_ring_joint
    // CHECK: "ttnn.scaled_dot_product_attention"
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>, tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x128x64xbf16, #sharded>
    return %2 : tensor<1x8x128x64xbf16, #sharded>
  }

  // The gather is on the head axis, not the sequence axis. Q carries the full
  // 16 heads so the SDPA itself stays valid.
  func.func @no_fire_wrong_gather_dim(%q: tensor<1x16x128x64xbf16, #heads>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x16x128x64xbf16, #heads> {
    // CHECK-LABEL: @no_fire_wrong_gather_dim
    // CHECK-NOT: exp_ring_joint
    // CHECK: "ttnn.scaled_dot_product_attention"
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x16x128x64xbf16, #heads>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x16x128x64xbf16, #heads>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x16x128x64xbf16, #heads>, tensor<1x16x128x64xbf16, #heads>, tensor<1x16x128x64xbf16, #heads>) -> tensor<1x16x128x64xbf16, #heads>
    return %2 : tensor<1x16x128x64xbf16, #heads>
  }

  // cluster_axis 0 spans a single device on a 1x2 mesh; a one-device ring is
  // not worth forming.
  func.func @no_fire_single_device_ring(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x128x64xbf16, #sharded> {
    // CHECK-LABEL: @no_fire_single_device_ring
    // CHECK-NOT: exp_ring_joint
    // CHECK: "ttnn.scaled_dot_product_attention"
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>, tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x128x64xbf16, #sharded>
    return %2 : tensor<1x8x128x64xbf16, #sharded>
  }

  // The K all-gather has a second consumer, so absorbing it is not free.
  func.func @no_fire_multi_use_gather(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>) {
    // CHECK-LABEL: @no_fire_multi_use_gather
    // CHECK-NOT: exp_ring_joint
    // CHECK: "ttnn.scaled_dot_product_attention"
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>, tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x128x64xbf16, #sharded>
    return %2, %0 : tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>
  }

  // Same shape the positive test uses: proves the OFF run is only gated by the
  // pass option, not by some accidental guard.
  func.func @would_fire_when_enabled(%q: tensor<1x8x128x64xbf16, #sharded>, %k: tensor<1x8x128x64xbf16, #sharded>, %v: tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x128x64xbf16, #sharded> {
    // CHECK-LABEL: @would_fire_when_enabled
    // CHECK: exp_ring_joint
    // OFF-LABEL: @would_fire_when_enabled
    // OFF: "ttnn.scaled_dot_product_attention"
    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>, tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x128x64xbf16, #sharded>
    return %2 : tensor<1x8x128x64xbf16, #sharded>
  }
}
