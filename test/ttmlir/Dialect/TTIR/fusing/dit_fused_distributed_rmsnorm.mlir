// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @wan_qk_rms_heads
  func.func @wan_qk_rms_heads(%arg0: tensor<1x4096x1280xbf16>, %arg1: tensor<1280xbf16>) -> tensor<1x10x4096x128xbf16> {
    // CHECK: %[[W:.*]] = "ttir.reshape"(%arg1)
    // CHECK-SAME: shape = [1 : i32, 1280 : i32]
    // CHECK: "ttcore.composite"(%arg0, %[[W]])
    // CHECK-SAME: num_heads_per_device = 10
    // CHECK-SAME: composite_name = "dit_fused_distributed_rmsnorm"
    // CHECK-NOT: ttir.distributed_rms_norm
    // CHECK-NOT: ttir.permute
    %0 = "ttir.distributed_rms_norm"(%arg0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x4096x1280xbf16>, tensor<1280xbf16>) -> tensor<1x4096x1280xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 4096 : i32, 10 : i32, 128 : i32]}> : (tensor<1x4096x1280xbf16>) -> tensor<1x4096x10x128xbf16>
    %2 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4096x10x128xbf16>) -> tensor<1x10x4096x128xbf16>
    return %2 : tensor<1x10x4096x128xbf16>
  }

  // Self-attn Q: RMS -> deinterleave -> half-rotation RoPE -> heads permute.
  // CHECK-LABEL: func.func @wan_self_attn_q_rms_rope_heads
  // The `cat(c, c)` widening is peeled off, so the composite takes the narrow
  // `head_dim/2` caches.
  // The caches are built *below* the norm, as in the real graph, so the
  // composite has to be emitted at the end of the subgraph to stay dominated.
  func.func @wan_self_attn_q_rms_rope_heads(
      %arg0: tensor<1x4096x1280xbf16>, %arg1: tensor<1280xbf16>,
      %cos_src: tensor<1x4096x1x128xbf16>, %sin_src: tensor<1x4096x1x128xbf16>)
      -> tensor<1x10x4096x128xbf16> {
    // CHECK: %[[COS:.*]] = "ttir.slice_static"(%arg2)
    // CHECK: %[[SIN:.*]] = "ttir.slice_static"(%arg3)
    // CHECK: "ttcore.composite"(%arg0, %{{[0-9a-z_]+}}, %[[COS]], %[[SIN]], %{{[0-9a-z_]+}})
    // CHECK-SAME: has_rope = true
    // CHECK-SAME: num_heads_per_device = 10
    // CHECK-SAME: composite_name = "dit_fused_distributed_rmsnorm"
    // CHECK-NOT: ttir.distributed_rms_norm
    // CHECK-NOT: ttir.neg
    %0 = "ttir.distributed_rms_norm"(%arg0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x4096x1280xbf16>, tensor<1280xbf16>) -> tensor<1x4096x1280xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 4096 : i32, 10 : i32, 64 : i32, 2 : i32]}> : (tensor<1x4096x1280xbf16>) -> tensor<1x4096x10x64x2xbf16>
    %2 = "ttir.permute"(%1) <{permutation = array<i64: 0, 1, 2, 4, 3>}> : (tensor<1x4096x10x64x2xbf16>) -> tensor<1x4096x10x2x64xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 4096 : i32, 10 : i32, 128 : i32]}> : (tensor<1x4096x10x2x64xbf16>) -> tensor<1x4096x10x128xbf16>
    %cos = "ttir.slice_static"(%cos_src) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4096 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<1x4096x1x128xbf16>) -> tensor<1x4096x1x64xbf16>
    %sin = "ttir.slice_static"(%sin_src) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4096 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<1x4096x1x128xbf16>) -> tensor<1x4096x1x64xbf16>
    %cos_full = "ttir.concat"(%cos, %cos) <{dim = 3 : si32}> : (tensor<1x4096x1x64xbf16>, tensor<1x4096x1x64xbf16>) -> tensor<1x4096x1x128xbf16>
    %sin_full = "ttir.concat"(%sin, %sin) <{dim = 3 : si32}> : (tensor<1x4096x1x64xbf16>, tensor<1x4096x1x64xbf16>) -> tensor<1x4096x1x128xbf16>
    %cos_bc = "ttir.broadcast"(%cos_full) <{broadcast_dimensions = array<i64: 1, 1, 10, 1>}> : (tensor<1x4096x1x128xbf16>) -> tensor<1x4096x10x128xbf16>
    %sin_bc = "ttir.broadcast"(%sin_full) <{broadcast_dimensions = array<i64: 1, 1, 10, 1>}> : (tensor<1x4096x1x128xbf16>) -> tensor<1x4096x10x128xbf16>
    %4 = "ttir.multiply"(%3, %cos_bc) : (tensor<1x4096x10x128xbf16>, tensor<1x4096x10x128xbf16>) -> tensor<1x4096x10x128xbf16>
    %5 = "ttir.slice_static"(%3) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4096 : i32, 10 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4096x10x128xbf16>) -> tensor<1x4096x10x64xbf16>
    %6 = "ttir.slice_static"(%3) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 4096 : i32, 10 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4096x10x128xbf16>) -> tensor<1x4096x10x64xbf16>
    %7 = "ttir.neg"(%6) : (tensor<1x4096x10x64xbf16>) -> tensor<1x4096x10x64xbf16>
    %8 = "ttir.concat"(%7, %5) <{dim = 3 : si32}> : (tensor<1x4096x10x64xbf16>, tensor<1x4096x10x64xbf16>) -> tensor<1x4096x10x128xbf16>
    %9 = "ttir.multiply"(%8, %sin_bc) : (tensor<1x4096x10x128xbf16>, tensor<1x4096x10x128xbf16>) -> tensor<1x4096x10x128xbf16>
    %10 = "ttir.add"(%4, %9) : (tensor<1x4096x10x128xbf16>, tensor<1x4096x10x128xbf16>) -> tensor<1x4096x10x128xbf16>
    %11 = "ttir.permute"(%10) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4096x10x128xbf16>) -> tensor<1x10x4096x128xbf16>
    return %11 : tensor<1x10x4096x128xbf16>
  }

  // CHECK-LABEL: func.func @no_fuse_without_heads
  func.func @no_fuse_without_heads(%arg0: tensor<1x4096x1280xbf16>, %arg1: tensor<1280xbf16>) -> tensor<1x4096x1280xbf16> {
    // CHECK: "ttir.distributed_rms_norm"
    // CHECK-NOT: "ttcore.composite"
    %0 = "ttir.distributed_rms_norm"(%arg0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x4096x1280xbf16>, tensor<1280xbf16>) -> tensor<1x4096x1280xbf16>
    return %0 : tensor<1x4096x1280xbf16>
  }
}
