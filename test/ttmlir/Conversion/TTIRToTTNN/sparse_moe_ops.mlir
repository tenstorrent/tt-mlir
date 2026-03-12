// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=2,4" %s | FileCheck %s
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sparse MoE ops TTIR to TTNN lowering test.
// Shapes extracted from a real GPT-OSS model dump with 2x4 mesh (post-sharding local shapes).

// Verify lowering of ttir.all_to_all_dispatch to ttnn.all_to_all_dispatch
module attributes {} {
  func.func @all_to_all_dispatch(
    %activations: tensor<1x1x128x2880xbf16>,
    %indices: tensor<1x1x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>) {
    %dispatched, %metadata = "ttir.all_to_all_dispatch"(%activations, %indices, %expert_mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %dispatched, %metadata : tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>
  }
}
// CHECK-LABEL: @all_to_all_dispatch
// CHECK: "ttnn.all_to_all_dispatch"
// CHECK-SAME: cluster_axis = 0
// CHECK-SAME: num_devices = 2

// -----

// Verify frontend-friendly dispatch ranks are canonicalized by MoE workarounds.
module attributes {} {
  func.func @all_to_all_dispatch_flexible(
    %activations: tensor<1x128x2880xbf16>,
    %indices: tensor<128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>) {
    %dispatched, %metadata = "ttir.all_to_all_dispatch"(%activations, %indices, %expert_mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<1x128x2880xbf16>, tensor<128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %dispatched, %metadata : tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>
  }
}
// CHECK-LABEL: @all_to_all_dispatch_flexible
// CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [1 : i32, 1 : i32, 128 : i32, 2880 : i32]}>
// CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [1 : i32, 1 : i32, 128 : i32, 4 : i32]}>
// CHECK: "ttnn.all_to_all_dispatch"

// -----

// Verify lowering of ttir.moe_expert_token_remap to ttnn.moe_expert_token_remap
module attributes {} {
  func.func @moe_expert_token_remap(
    %topk_weights: tensor<1x2x128x32xbf16>,
    %expert_mapping: tensor<1x1x32x8xi64>,
    %metadata: tensor<1x2x128x4xi64>
  ) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>) {
    %mapping, %reduced = "ttir.moe_expert_token_remap"(%topk_weights, %expert_mapping, %metadata) <{reduction_size = 32 : i64}> : (tensor<1x2x128x32xbf16>, tensor<1x1x32x8xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)
    return %mapping, %reduced : tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>
  }
}
// CHECK-LABEL: @moe_expert_token_remap
// CHECK: "ttnn.moe_expert_token_remap"
// CHECK-SAME: reduction_size = 32

// -----

// Verify frontend-friendly remap topk ranks are canonicalized by MoE workarounds.
module attributes {} {
  func.func @moe_expert_token_remap_flexible_topk(
    %topk_weights: tensor<128x32xbf16>,
    %expert_mapping: tensor<1x1x32x8xi64>,
    %metadata: tensor<1x2x128x4xi64>
  ) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>) {
    %mapping, %reduced = "ttir.moe_expert_token_remap"(%topk_weights, %expert_mapping, %metadata) <{reduction_size = 32 : i64}> : (tensor<128x32xbf16>, tensor<1x1x32x8xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)
    return %mapping, %reduced : tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>
  }
}
// CHECK-LABEL: @moe_expert_token_remap_flexible_topk
// CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [1 : i32, 128 : i32, 32 : i32]}>
// CHECK: "ttnn.concat"(%{{.*}}, %{{.*}})
// CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [1 : i32, 2 : i32, 128 : i32, 32 : i32]}>
// CHECK: "ttnn.moe_expert_token_remap"

// -----

// Verify lowering of ttir.sparse_matmul (gate_up variant) to ttnn.sparse_matmul
module attributes {} {
  func.func @sparse_matmul_gate_up(
    %input: tensor<2x4x32x2880xbf16>,
    %weight: tensor<1x4x2880x5760xbf16>,
    %sparsity: tensor<2x4x1x4xbf16>
  ) -> tensor<2x4x1x4x32x5760xbf16> {
    %result = "ttir.sparse_matmul"(%input, %weight, %sparsity) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
    return %result : tensor<2x4x1x4x32x5760xbf16>
  }
}
// CHECK-LABEL: @sparse_matmul_gate_up
// CHECK: "ttnn.sparse_matmul"
// CHECK-SAME: is_input_a_sparse = false
// CHECK-SAME: is_input_b_sparse = true

// -----

// Verify lowering of ttir.sparse_matmul (down variant) to ttnn.sparse_matmul
module attributes {} {
  func.func @sparse_matmul_down(
    %input: tensor<8x4x32x2880xbf16>,
    %weight: tensor<1x4x2880x2880xbf16>,
    %sparsity: tensor<1x1x8x4xbf16>
  ) -> tensor<8x4x32x2880xbf16> {
    %result = "ttir.sparse_matmul"(%input, %weight, %sparsity) <{is_input_a_sparse = true, is_input_b_sparse = false, nnz = 0 : i64}> : (tensor<8x4x32x2880xbf16>, tensor<1x4x2880x2880xbf16>, tensor<1x1x8x4xbf16>) -> tensor<8x4x32x2880xbf16>
    return %result : tensor<8x4x32x2880xbf16>
  }
}
// CHECK-LABEL: @sparse_matmul_down
// CHECK: "ttnn.sparse_matmul"
// CHECK-SAME: is_input_a_sparse = true
// CHECK-SAME: is_input_b_sparse = false

// -----

// Verify lowering of ttir.all_to_all_combine to ttnn.all_to_all_combine
module attributes {} {
  func.func @all_to_all_combine(
    %input: tensor<4x2x128x2880xbf16>,
    %metadata: tensor<1x2x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> tensor<4x1x128x2880xbf16> {
    %result = "ttir.all_to_all_combine"(%input, %metadata, %expert_mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %result : tensor<4x1x128x2880xbf16>
  }
}
// CHECK-LABEL: @all_to_all_combine
// CHECK: "ttnn.all_to_all_combine"
// CHECK-SAME: cluster_axis = 0
// CHECK-SAME: num_devices = 2
// CHECK-SAME: num_experts_per_tok = 4

// -----

// Verify frontend-friendly combine input layout [BD, S, E, H] is canonicalized
// to [E, BD, S, H] by MoE workarounds.
module attributes {} {
  func.func @all_to_all_combine_flexible_input_layout(
    %input: tensor<2x128x4x2880xbf16>,
    %metadata: tensor<1x2x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> tensor<4x1x128x2880xbf16> {
    %result = "ttir.all_to_all_combine"(%input, %metadata, %expert_mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64}> : (tensor<2x128x4x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %result : tensor<4x1x128x2880xbf16>
  }
}
// CHECK-LABEL: @all_to_all_combine_flexible_input_layout
// CHECK: "ttnn.permute"(%{{.*}}) <{permutation = array<i64: 2, 0, 1, 3>}>
// CHECK: "ttnn.all_to_all_combine"

// -----

// Verify output_shard_dim is forwarded from TTIR to TTNN.
module attributes {} {
  func.func @all_to_all_combine_forward_output_shard_dim(
    %input: tensor<4x2x128x2880xbf16>,
    %metadata: tensor<1x2x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> tensor<4x1x128x2880xbf16> {
    %result = "ttir.all_to_all_combine"(%input, %metadata, %expert_mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64, output_shard_dim = 2 : i64}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %result : tensor<4x1x128x2880xbf16>
  }
}
// CHECK-LABEL: @all_to_all_combine_forward_output_shard_dim
// CHECK: "ttnn.all_to_all_combine"
// CHECK-SAME: output_shard_dim = 2
