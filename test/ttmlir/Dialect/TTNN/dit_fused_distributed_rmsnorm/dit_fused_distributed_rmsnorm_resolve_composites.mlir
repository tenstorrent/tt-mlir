// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttnn-resolve-composites="composite-resolution=force-promote" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @wan_qk_rms_heads
  func.func @wan_qk_rms_heads(%arg0: tensor<1x4096x1280xbf16>, %arg1: tensor<1x1280xbf16>) -> tensor<1x10x4096x128xbf16> {
    // CHECK: "ttnn.get_device"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 4096 : i32, 1280 : i32]
    // CHECK: "ttnn.dit_fused_distributed_rmsnorm"
    // CHECK-SAME: cluster_axis = 1
    // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>
    // CHECK-SAME: num_heads_per_device = 10
    // CHECK-SAME: num_links = 1
    // CHECK-NOT: "ttnn.reshape"
    // CHECK-NOT: "ttcore.composite"
    // CHECK-NOT: @dit_fused_distributed_rmsnorm_decomp
    %0 = "ttcore.composite"(%arg0, %arg1)
        <{composite_name = "dit_fused_distributed_rmsnorm",
          decomposition = @dit_fused_distributed_rmsnorm_decomp,
          composite_attributes = {cluster_axis = 1 : i32, epsilon = 1.000000e-05 : f32, num_heads_per_device = 10 : i32, per_head_norm = false, has_bias = false, has_rope = false}}>
        : (tensor<1x4096x1280xbf16>, tensor<1x1280xbf16>) -> tensor<1x10x4096x128xbf16>
    return %0 : tensor<1x10x4096x128xbf16>
  }

  func.func private @dit_fused_distributed_rmsnorm_decomp(
      %arg0: tensor<1x4096x1280xbf16>,
      %arg1: tensor<1x1280xbf16>
  ) -> tensor<1x10x4096x128xbf16> {
    %w = "ttir.reshape"(%arg1) <{shape = [1280 : i32]}> : (tensor<1x1280xbf16>) -> tensor<1280xbf16>
    %0 = "ttir.distributed_rms_norm"(%arg0, %w) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x4096x1280xbf16>, tensor<1280xbf16>) -> tensor<1x4096x1280xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 4096 : i32, 10 : i32, 128 : i32]}> : (tensor<1x4096x1280xbf16>) -> tensor<1x4096x10x128xbf16>
    %2 = "ttir.permute"(%1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4096x10x128xbf16>) -> tensor<1x10x4096x128xbf16>
    return %2 : tensor<1x10x4096x128xbf16>
  }
}
