// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=2,4" %s | FileCheck %s
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sparse MoE ops TTIR to TTNN lowering test.
// Shapes extracted from a real GPT-OSS model dump with 2x4 mesh (post-sharding local shapes).

// Verify lowering of ttir.composite "tt.all_to_all_dispatch" to ttnn.all_to_all_dispatch
module attributes {} {
  func.func @all_to_all_dispatch(
    %activations: tensor<1x1x128x2880xbf16>,
    %indices: tensor<1x1x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>) {
    %dispatched, %metadata = "ttir.composite"(%activations, %indices, %expert_mapping) <{name = "tt.all_to_all_dispatch", op_attributes = {cluster_axis = 0 : i64, num_devices = 2 : i64}}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %dispatched, %metadata : tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>
  }
}
// CHECK-LABEL: @all_to_all_dispatch
// CHECK: "ttnn.all_to_all_dispatch"
// CHECK-SAME: cluster_axis = 0
// CHECK-SAME: num_devices = 2

// -----

// Verify lowering of ttir.composite "tt.all_to_all_dispatch_metadata" to ttnn.all_to_all_dispatch_metadata
module attributes {} {
  func.func @all_to_all_dispatch_metadata(
    %activations: tensor<1x1x128x2880xbf16>,
    %indices: tensor<1x1x128x4xi64>,
    %scores: tensor<1x1x128x4xbf16>,
    %expert_mapping: tensor<1x1x8x32xi64>
  ) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>) {
    %dispatched, %idx_out, %scores_out = "ttir.composite"(%activations, %indices, %scores, %expert_mapping) <{name = "tt.all_to_all_dispatch_metadata", op_attributes = {cluster_axis = 0 : i64, num_devices = 2 : i64}}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x8x32xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>)
    return %dispatched, %idx_out, %scores_out : tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>
  }
}
// CHECK-LABEL: @all_to_all_dispatch_metadata
// CHECK: "ttnn.all_to_all_dispatch_metadata"
// CHECK-SAME: cluster_axis = 0
// CHECK-SAME: drain_core = #ttnn.core_coord<0, 0>
// CHECK-SAME: num_devices = 2

// -----

// Verify lowering of ttir.composite "tt.moe_expert_token_remap" to ttnn.moe_expert_token_remap
module attributes {} {
  func.func @moe_expert_token_remap(
    %topk_weights: tensor<1x2x128x32xbf16>,
    %expert_mapping: tensor<1x1x32x8xi64>,
    %metadata: tensor<1x2x128x4xi64>
  ) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>) {
    %mapping, %reduced = "ttir.composite"(%topk_weights, %expert_mapping, %metadata) <{name = "tt.moe_expert_token_remap", op_attributes = {reduction_size = 32 : i64}}> : (tensor<1x2x128x32xbf16>, tensor<1x1x32x8xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)
    return %mapping, %reduced : tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>
  }
}
// CHECK-LABEL: @moe_expert_token_remap
// CHECK: "ttnn.moe_expert_token_remap"
// CHECK-SAME: reduction_size = 32

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

// Verify lowering of ttir.composite "tt.all_to_all_combine" to ttnn.all_to_all_combine
module attributes {} {
  func.func @all_to_all_combine(
    %input: tensor<4x2x128x2880xbf16>,
    %metadata: tensor<1x2x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> tensor<4x1x128x2880xbf16> {
    %result = "ttir.composite"(%input, %metadata, %expert_mapping) <{name = "tt.all_to_all_combine", op_attributes = {cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64, output_shard_dim = 1 : i64}}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %result : tensor<4x1x128x2880xbf16>
  }
}
// CHECK-LABEL: @all_to_all_combine
// CHECK: "ttnn.all_to_all_combine"
// CHECK-SAME: cluster_axis = 0
// CHECK-SAME: num_devices = 2
// CHECK-SAME: num_experts_per_tok = 4
// CHECK-SAME: output_shard_dim = 1

// -----

// Verify output_shard_dim=2 is passed through from TTIR to TTNN
module attributes {} {
  func.func @all_to_all_combine_shard_dim_2(
    %input: tensor<4x1x128x2880xbf16>,
    %metadata: tensor<1x1x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> tensor<4x1x64x2880xbf16> {
    %result = "ttir.composite"(%input, %metadata, %expert_mapping) <{name = "tt.all_to_all_combine", op_attributes = {cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64, output_shard_dim = 2 : i64}}> : (tensor<4x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x64x2880xbf16>
    return %result : tensor<4x1x64x2880xbf16>
  }
}
// CHECK-LABEL: @all_to_all_combine_shard_dim_2
// CHECK: "ttnn.all_to_all_combine"
// CHECK-SAME: output_shard_dim = 2
