// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline %s | FileCheck %s

// stablehlo.custom_call @tt.moe_decode -> ttcore.composite "moe_decode" + a
// synthesized private decomposition (all_gather weights -> two sparse_matmuls +
// GLU -> one-hot top-k select). The frontend composite is mesh-agnostic:
// num_devices is derived from the module mesh (num_devices =
// meshShape[cluster_axis]) and the expert_mapping is synthesized here.

// Single-axis mesh: the combine already reduces the whole mesh, so no trailing
// all_reduce is inserted.
module @moe_decode_single_axis attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x8>]>} {
  func.func public @main(
      %tokens: tensor<1x1x32x64xbf16>,
      %indices: tensor<1x1x32x2xui16>,
      %scores: tensor<1x1x32x2xbf16>,
      %w0: tensor<1x1x64x64xbf16>,
      %w1: tensor<1x1x64x64xbf16>,
      %w2: tensor<1x1x64x64xbf16>) -> tensor<2x32x64xbf16> {
    // CHECK-LABEL: @main
    // The synthesized expert_mapping constant is re-inserted at operand 3.
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2, %{{[0-9]+}}, %arg3, %arg4, %arg5)
    // CHECK-SAME: cluster_axis = 1 : i64
    // CHECK-SAME: composite_name = "moe_decode"
    // CHECK-SAME: decomposition = @moe_decode_decomp
    // No compound sharding on a single-axis mesh.
    // CHECK-NOT: "ttir.all_reduce"
    %0 = stablehlo.custom_call @tt.moe_decode(%tokens, %indices, %scores, %w0, %w1, %w2) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", layer_id = "0", output_height_shard_dim = "4", intermediate_size = "64", activation_function = "silu"}} : (tensor<1x1x32x64xbf16>, tensor<1x1x32x2xui16>, tensor<1x1x32x2xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<2x32x64xbf16>
    return %0 : tensor<2x32x64xbf16>
  }
  // The decomposition holds the primitive reference lowering.
  // CHECK: func.func private @moe_decode_decomp
  // CHECK: "ttir.all_gather"
  // CHECK: "ttir.sparse_matmul"
  // CHECK: "ttir.silu"
  // CHECK: "ttir.matmul"
}

// -----

// Compound sharding: a 2x4 mesh with cluster_axis=1 leaves 2 non-cluster
// devices each holding a partial, so an all_reduce(sum) over the non-cluster
// axis (0) aggregates the combine output. bh_ring_size rides along on the
// composite as (currently unconsumed) metadata.
module @moe_decode_compound attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  func.func public @main(
      %tokens: tensor<1x1x32x64xbf16>,
      %indices: tensor<1x1x32x2xui16>,
      %scores: tensor<1x1x32x2xbf16>,
      %w0: tensor<1x1x64x64xbf16>,
      %w1: tensor<1x1x64x64xbf16>,
      %w2: tensor<1x1x64x64xbf16>) -> tensor<2x32x64xbf16> {
    // CHECK-LABEL: @main
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2, %{{[0-9]+}}, %arg3, %arg4, %arg5)
    // CHECK-SAME: bh_ring_size = 8 : i64
    // CHECK-SAME: composite_name = "moe_decode"
    // The trailing reduce over the non-cluster axis aggregates the partials.
    // CHECK: "ttir.all_reduce"
    %0 = stablehlo.custom_call @tt.moe_decode(%tokens, %indices, %scores, %w0, %w1, %w2) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", layer_id = "0", output_height_shard_dim = "4", intermediate_size = "64", activation_function = "silu", bh_ring_size = "8"}} : (tensor<1x1x32x64xbf16>, tensor<1x1x32x2xui16>, tensor<1x1x32x2xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<2x32x64xbf16>
    return %0 : tensor<2x32x64xbf16>
  }
}
