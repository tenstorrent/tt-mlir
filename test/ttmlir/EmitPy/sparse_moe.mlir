// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --ttir-to-emitpy-pipeline="mesh-shape=2,4" %s | FileCheck %s

// Verify EmitPy codegen for sparse MoE ops.

// CHECK: emitpy.call_opaque "ttnn.all_to_all_dispatch"
module attributes {} {
  func.func @all_to_all_dispatch(
    %activations: tensor<1x1x128x2880xbf16>,
    %indices: tensor<1x1x128x4xi64>,
    %expert_mapping: tensor<1x1x8x32xi64>
  ) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>) {
    %dispatched, %metadata = "ttir.all_to_all_dispatch"(%activations, %indices, %expert_mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x8x32xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %dispatched, %metadata : tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>
  }
}

// -----

// CHECK: emitpy.call_opaque "ttnn.experimental.all_to_all_dispatch_metadata"
module attributes {} {
  func.func @all_to_all_dispatch_metadata(
    %activations: tensor<1x1x128x2880xbf16>,
    %indices: tensor<1x1x128x4xi64>,
    %scores: tensor<1x1x128x4xbf16>,
    %expert_mapping: tensor<1x1x8x32xi64>
  ) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>) {
    %dispatched, %idx_out, %scores_out = "ttir.all_to_all_dispatch_metadata"(%activations, %indices, %scores, %expert_mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x128x4xbf16>, tensor<1x1x8x32xi64>) -> (tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>)
    return %dispatched, %idx_out, %scores_out : tensor<1x256x2880xbf16>, tensor<1x256x4xi64>, tensor<1x256x4xbf16>
  }
}

// -----

// CHECK: emitpy.call_opaque "ttnn.moe_expert_token_remap"
module attributes {} {
  func.func @moe_expert_token_remap(
    %topk_weights: tensor<1x2x128x32xbf16>,
    %expert_mapping: tensor<1x1x8x32xi64>,
    %metadata: tensor<1x2x128x4xi64>
  ) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>) {
    %mapping, %reduced = "ttir.moe_expert_token_remap"(%topk_weights, %expert_mapping, %metadata) <{reduction_size = 32 : i64}> : (tensor<1x2x128x32xbf16>, tensor<1x1x8x32xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)
    return %mapping, %reduced : tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>
  }
}

// -----

// CHECK: emitpy.call_opaque "ttnn.sparse_matmul"
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

// -----

// CHECK: emitpy.call_opaque "ttnn.sparse_matmul"
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

// -----

// CHECK: emitpy.call_opaque "ttnn.all_to_all_combine"
module attributes {} {
  func.func @all_to_all_combine(
    %input: tensor<4x2x128x2880xbf16>,
    %metadata: tensor<1x2x128x4xi64>,
    %expert_mapping: tensor<1x1x8x32xi64>
  ) -> tensor<4x1x128x2880xbf16> {
    %result = "ttir.all_to_all_combine"(%input, %metadata, %expert_mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x8x32xi64>) -> tensor<4x1x128x2880xbf16>
    return %result : tensor<4x1x128x2880xbf16>
  }
}
