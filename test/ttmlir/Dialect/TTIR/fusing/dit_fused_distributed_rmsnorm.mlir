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

  // CHECK-LABEL: func.func @no_fuse_without_heads
  func.func @no_fuse_without_heads(%arg0: tensor<1x4096x1280xbf16>, %arg1: tensor<1280xbf16>) -> tensor<1x4096x1280xbf16> {
    // CHECK: "ttir.distributed_rms_norm"
    // CHECK-NOT: "ttcore.composite"
    %0 = "ttir.distributed_rms_norm"(%arg0, %arg1) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x4096x1280xbf16>, tensor<1280xbf16>) -> tensor<1x4096x1280xbf16>
    return %0 : tensor<1x4096x1280xbf16>
  }
}
