// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --ttnn-workaround --canonicalize %s | FileCheck %s

// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test bundled MoE workarounds.
// Frontend-friendly rank/layout forms should be canonicalized here, not in
// conversion patterns.
// Decode-like all_to_all_combine with output_shard_dim=2 and S=1 should be
// rewritten to output_shard_dim=1 + reshape back to the original result shape.

module @test_moe_workaround attributes {} {

  func.func public @all_to_all_dispatch_flexible(
    %input: tensor<64x128x7168xbf16>,
    %indices: tensor<8192x8xi64>,
    %mapping: tensor<1x1x256x8xi64>
  ) -> (tensor<1x64x128x7168xbf16>, tensor<1x64x128x8xi64>) {
    // CHECK-LABEL: func.func public @all_to_all_dispatch_flexible
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [64 : i32, 1 : i32, 128 : i32, 7168 : i32]}>
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [64 : i32, 1 : i32, 128 : i32, 8 : i32]}>
    // CHECK: "ttnn.all_to_all_dispatch"(%{{.*}}, %{{.*}}, %{{.*}})
    %dispatched, %metadata = "ttnn.all_to_all_dispatch"(%input, %indices, %mapping) <{
      cluster_axis = 0 : i64,
      num_devices = 2 : i64
    }> : (tensor<64x128x7168xbf16>, tensor<8192x8xi64>, tensor<1x1x256x8xi64>) -> (tensor<1x64x128x7168xbf16>, tensor<1x64x128x8xi64>)
    return %dispatched, %metadata : tensor<1x64x128x7168xbf16>, tensor<1x64x128x8xi64>
  }

  func.func public @all_to_all_combine_layout_flexible(
    %input: tensor<64x128x32x7168xbf16>,
    %metadata: tensor<1x64x128x8xi64>,
    %mapping: tensor<1x1x256x8xi64>
  ) -> tensor<8x8x128x7168xbf16> {
    // CHECK-LABEL: func.func public @all_to_all_combine_layout_flexible
    // CHECK: "ttnn.permute"(%{{.*}}) <{permutation = array<i64: 2, 0, 1, 3>}>
    // CHECK: "ttnn.all_to_all_combine"(%{{.*}}, %{{.*}}, %{{.*}})
    %result = "ttnn.all_to_all_combine"(%input, %metadata, %mapping) <{
      cluster_axis = 0 : i64,
      num_devices = 8 : i64,
      num_experts_per_tok = 8 : i64,
      output_shard_dim = 1 : i64
    }> : (tensor<64x128x32x7168xbf16>, tensor<1x64x128x8xi64>, tensor<1x1x256x8xi64>) -> tensor<8x8x128x7168xbf16>
    return %result : tensor<8x8x128x7168xbf16>
  }

  func.func public @all_to_all_combine_layout_single_expert_prefill(
    %input: tensor<4x128x1x2880xbf16>,
    %metadata: tensor<1x4x128x4xi64>,
    %mapping: tensor<1x1x32x32xi64>
  ) -> tensor<4x1x128x2880xbf16> {
    // CHECK-LABEL: func.func public @all_to_all_combine_layout_single_expert_prefill
    // CHECK: "ttnn.permute"(%{{.*}}) <{permutation = array<i64: 2, 0, 1, 3>}>
    // CHECK: "ttnn.all_to_all_combine"(%{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-SAME: output_shard_dim = 1 : i64
    %result = "ttnn.all_to_all_combine"(%input, %metadata, %mapping) <{
      cluster_axis = 0 : i64,
      num_devices = 4 : i64,
      num_experts_per_tok = 4 : i64,
      output_shard_dim = 2 : i64
    }> : (tensor<4x128x1x2880xbf16>, tensor<1x4x128x4xi64>, tensor<1x1x32x32xi64>) -> tensor<4x1x128x2880xbf16>
    return %result : tensor<4x1x128x2880xbf16>
  }

  func.func public @moe_expert_token_remap_flexible(
    %topk: tensor<1024x8xbf16>,
    %mapping: tensor<1x1x256x8xi64>,
    %metadata: tensor<1x64x128x8xi64>
  ) -> (tensor<1x64x128x8xbf16>, tensor<1x1x8x8xbf16>) {
    // CHECK-LABEL: func.func public @moe_expert_token_remap_flexible
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [8 : i32, 128 : i32, 8 : i32]}>
    // CHECK: "ttnn.concat"(%{{.*}}
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [1 : i32, 64 : i32, 128 : i32, 8 : i32]}>
    // CHECK: "ttnn.moe_expert_token_remap"(%{{.*}}, %{{.*}}, %{{.*}})
    %mapping_out, %reduced = "ttnn.moe_expert_token_remap"(%topk, %mapping, %metadata) <{
      reduction_size = 8 : i64
    }> : (tensor<1024x8xbf16>, tensor<1x1x256x8xi64>, tensor<1x64x128x8xi64>) -> (tensor<1x64x128x8xbf16>, tensor<1x1x8x8xbf16>)
    return %mapping_out, %reduced : tensor<1x64x128x8xbf16>, tensor<1x1x8x8xbf16>
  }

  func.func public @all_to_all_combine_decode(
    %input: tensor<8x64x1x7168xbf16>,
    %metadata: tensor<1x64x1x8xi64>,
    %mapping: tensor<1x1x256x8xi64>
  ) -> tensor<8x8x1x7168xbf16> {
    // CHECK-LABEL: func.func public @all_to_all_combine_decode
    // CHECK: "ttnn.all_to_all_combine"
    // CHECK-SAME: output_shard_dim = 1
    // CHECK-SAME: -> tensor<8x1x8x7168xbf16
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [8 : i32, 8 : i32, 1 : i32, 7168 : i32]}>
    %result = "ttnn.all_to_all_combine"(%input, %metadata, %mapping) <{
      cluster_axis = 0 : i64,
      num_devices = 2 : i64,
      num_experts_per_tok = 8 : i64,
      output_shard_dim = 2 : i64
    }> : (tensor<8x64x1x7168xbf16>, tensor<1x64x1x8xi64>, tensor<1x1x256x8xi64>) -> tensor<8x8x1x7168xbf16>
    return %result : tensor<8x8x1x7168xbf16>
  }
}
