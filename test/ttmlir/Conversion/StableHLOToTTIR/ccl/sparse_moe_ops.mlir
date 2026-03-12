// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline --stablehlo-to-ttir-pipeline %s | FileCheck %s

// Sparse MoE custom ops: all_to_all_dispatch, moe_expert_token_remap,
// sparse_matmul (gate_up and down), and all_to_all_combine.
// Shapes extracted from a real GPT-OSS model dump with 2x4 mesh (post-sharding).

module @sparse_moe_ops attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func public @main(
    %activations: tensor<1x1x128x2880xbf16>,
    %indices: tensor<1x1x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>,
    %topk_weights: tensor<1x2x128x32xbf16>,
    %gate_up_weight: tensor<1x4x2880x5760xbf16>,
    %down_weight: tensor<1x4x2880x2880xbf16>
  ) -> tensor<4x1x128x2880xbf16> {
    %0 = sdy.manual_computation(%activations, %indices, %expert_mapping, %topk_weights, %gate_up_weight, %down_weight)
      in_shardings=[
        <@mesh, [{}, {}, {}, {}]>,
        <@mesh, [{}, {}, {}, {}]>,
        <@mesh, [{}, {}, {}, {}]>,
        <@mesh, [{}, {}, {}, {}]>,
        <@mesh, [{}, {}, {}, {}]>,
        <@mesh, [{}, {}, {}, {}]>
      ]
      out_shardings=[<@mesh, [{}, {}, {}, {}]>]
      manual_axes={"_axis_0", "_axis_1"}
    (
      %arg0: tensor<1x1x128x2880xbf16>,
      %arg1: tensor<1x1x128x4xi64>,
      %arg2: tensor<1x1x32x8xi64>,
      %arg3: tensor<1x2x128x32xbf16>,
      %arg4: tensor<1x4x2880x5760xbf16>,
      %arg5: tensor<1x4x2880x2880xbf16>
    ) {
      // 1. all_to_all_dispatch
      %dispatch:2 = stablehlo.custom_call @tt.all_to_all_dispatch(%arg0, %arg1, %arg2) {
        api_version = 0 : i32,
        mhlo.frontend_attributes = {cluster_axis = "0", num_devices = "2"}
      } : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)

      // 2. moe_expert_token_remap
      %remap:2 = stablehlo.custom_call @tt.moe_expert_token_remap(%arg3, %arg2, %dispatch#1) {
        api_version = 0 : i32,
        mhlo.frontend_attributes = {reduction_size = "32"}
      } : (tensor<1x2x128x32xbf16>, tensor<1x1x32x8xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)

      // 3. Reshape for gate_up sparse_matmul
      %reshaped_act = stablehlo.reshape %dispatch#0 : (tensor<1x2x128x2880xbf16>) -> tensor<2x4x32x2880xbf16>
      %reshaped_reduced = stablehlo.reshape %remap#1 : (tensor<1x1x8x4xbf16>) -> tensor<2x4x1x4xbf16>

      // 4. sparse_matmul gate_up
      %gate_up = stablehlo.custom_call @tt.sparse_matmul(%reshaped_act, %arg4, %reshaped_reduced) {
        api_version = 0 : i32,
        mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True", nnz = "0"}
      } : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>

      // 5. Reshape + slice for down projection input
      %gate_up_reshaped = stablehlo.reshape %gate_up : (tensor<2x4x1x4x32x5760xbf16>) -> tensor<8x4x32x5760xbf16>
      %down_input = stablehlo.slice %gate_up_reshaped [0:8, 0:4, 0:32, 0:2880] : (tensor<8x4x32x5760xbf16>) -> tensor<8x4x32x2880xbf16>

      // 6. sparse_matmul down
      %down = stablehlo.custom_call @tt.sparse_matmul(%down_input, %arg5, %remap#1) {
        api_version = 0 : i32,
        mhlo.frontend_attributes = {is_input_a_sparse = "True", is_input_b_sparse = "False", nnz = "0"}
      } : (tensor<8x4x32x2880xbf16>, tensor<1x4x2880x2880xbf16>, tensor<1x1x8x4xbf16>) -> tensor<8x4x32x2880xbf16>

      // 7. Reshape for combine
      %down_reshaped = stablehlo.reshape %down : (tensor<8x4x32x2880xbf16>) -> tensor<4x2x128x2880xbf16>

      // 8. all_to_all_combine (uses sparse_matmul output → no DCE)
      %combine = stablehlo.custom_call @tt.all_to_all_combine(%down_reshaped, %dispatch#1, %arg2) {
        api_version = 0 : i32,
        mhlo.frontend_attributes = {cluster_axis = "0", num_devices = "2", num_experts_per_tok = "4", output_shard_dim = "1"}
      } : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>

      sdy.return %combine : tensor<4x1x128x2880xbf16>
    } : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>, tensor<1x2x128x32xbf16>, tensor<1x4x2880x5760xbf16>, tensor<1x4x2880x2880xbf16>) -> tensor<4x1x128x2880xbf16>
    return %0 : tensor<4x1x128x2880xbf16>
  }

  // Flexible frontend inputs:
  // [B, S, H] + [B*S, K] are preserved in TTIR and normalized in TTIR->TTNN.
  func.func public @dispatch_flexible_inputs(
    %activations_3d: tensor<1x128x2880xbf16>,
    %indices_2d: tensor<128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> tensor<1x2x128x2880xbf16> {
    %dispatch:2 = stablehlo.custom_call @tt.all_to_all_dispatch(%activations_3d, %indices_2d, %expert_mapping) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {cluster_axis = "0", num_devices = "2"}
    } : (tensor<1x128x2880xbf16>, tensor<128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
    return %dispatch#0 : tensor<1x2x128x2880xbf16>
  }

  // Flexible frontend combine input:
  // [BD, S, E, H] is preserved in TTIR and normalized in TTIR->TTNN.
  func.func public @combine_flexible_input_layout(
    %combine_input_bdseh: tensor<2x128x4x2880xbf16>,
    %metadata: tensor<1x2x128x4xi64>,
    %expert_mapping: tensor<1x1x32x8xi64>
  ) -> tensor<4x1x128x2880xbf16> {
    %combine = stablehlo.custom_call @tt.all_to_all_combine(%combine_input_bdseh, %metadata, %expert_mapping) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {cluster_axis = "0", num_devices = "2", num_experts_per_tok = "4"}
    } : (tensor<2x128x4x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %combine : tensor<4x1x128x2880xbf16>
  }

  // Flexible frontend topk input:
  // [B*S, E] is preserved in TTIR and normalized in TTIR->TTNN.
  func.func public @remap_flexible_topk(
    %topk_2d: tensor<128x32xbf16>,
    %expert_mapping: tensor<1x1x32x8xi64>,
    %metadata: tensor<1x2x128x4xi64>
  ) -> tensor<1x2x128x32xbf16> {
    %remap:2 = stablehlo.custom_call @tt.moe_expert_token_remap(%topk_2d, %expert_mapping, %metadata) {
      api_version = 0 : i32,
      mhlo.frontend_attributes = {num_devices = "2", reduction_size = "32"}
    } : (tensor<128x32xbf16>, tensor<1x1x32x8xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x32xbf16>, tensor<1x1x8x32xbf16>)
    return %remap#0 : tensor<1x2x128x32xbf16>
  }
}

// CHECK: "ttir.all_to_all_dispatch"
// CHECK-SAME: cluster_axis = 0
// CHECK-SAME: num_devices = 2

// CHECK: "ttir.moe_expert_token_remap"
// CHECK-SAME: reduction_size = 32

// CHECK: "ttir.sparse_matmul"
// CHECK-SAME: is_input_a_sparse = false
// CHECK-SAME: is_input_b_sparse = true

// CHECK: "ttir.sparse_matmul"
// CHECK-SAME: is_input_a_sparse = true
// CHECK-SAME: is_input_b_sparse = false

// CHECK: "ttir.all_to_all_combine"
// CHECK-SAME: cluster_axis = 0
// CHECK-SAME: num_devices = 2
// CHECK-SAME: num_experts_per_tok = 4
// CHECK-SAME: output_shard_dim = 1

// CHECK-LABEL: func.func public @dispatch_flexible_inputs
// CHECK: "ttir.all_to_all_dispatch"

// CHECK-LABEL: func.func public @combine_flexible_input_layout
// CHECK: "ttir.all_to_all_combine"

// CHECK-LABEL: func.func public @remap_flexible_topk
// CHECK: "ttir.moe_expert_token_remap"
