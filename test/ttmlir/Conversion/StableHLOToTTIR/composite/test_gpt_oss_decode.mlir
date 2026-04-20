// REQUIRES: stablehlo
// RUN: ttmlir-opt --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// The moe_gpt_decode composite uses a 4D layout at the composite boundary
// ([B, 1, S, H] for hidden/topk and [K, S, B, H] for the combined result) so
// it can directly bridge the surrounding MoE custom calls
// (tt.all_to_all_dispatch_metadata / tt.moe_gpt / tt.selective_reduce_combine)
// without any reshape at the boundary.
//
// The tt-metal moe_gpt kernel consumes only the fused 6D weight layout, so
// the composite's operand list is exactly 7 entries:
//   hidden, topk_indices, topk_scores, dispatch_mapping, moe_gpt_mapping,
//   fused_w0_w1, fused_w2
// The unfused gate_up_proj / down_proj experts are not part of this op — a
// separate non-composite path handles prefill.
module @gpt_oss_decode_composite_tests attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @moe_gpt_decode
  // CHECK: %[[RESULT:.*]] = "ttir.moe_gpt_decode"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
  // CHECK-SAME: alpha =
  // CHECK-SAME: cluster_axis = 1 : i64
  // CHECK-SAME: intermediate_size = 128 : i64
  // CHECK-SAME: limit =
  // CHECK-SAME: num_devices = 2 : i64
  // CHECK-SAME: num_experts = 8 : i64
  // CHECK-SAME: num_experts_per_tok = 2 : i64
  // CHECK-NOT: operandSegmentSizes
  // The fused weights follow tt-metal's `create_fused_moe_gpt_config` layout:
  //   w0_w1: [C_dram, L, E_local, 4, K+32, 4*TILE]
  //   w2:    [C_dram, L, E_local, 2, N+32, 4*TILE]
  func.func @moe_gpt_decode(%arg0: tensor<4x1x1x128xbf16>, %arg1: tensor<4x1x1x2xi64>, %arg2: tensor<4x1x1x2xbf16>, %arg3: tensor<1x1x2x8xi64>, %arg4: tensor<1x1x2x8xi64>, %arg5: tensor<12x1x8x4x160x128xbf16>, %arg6: tensor<12x1x8x2x160x128xbf16>) -> tensor<2x1x4x128xbf16> {
    // CHECK: return %[[RESULT]] : tensor<2x1x4x128xbf16>
    %0 = stablehlo.composite "tenstorrent.moe_gpt_decode" %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 {composite_attributes = {num_devices = 2 : i64, cluster_axis = 1 : i64, num_experts = 8 : i64, num_experts_per_tok = 2 : i64, intermediate_size = 128 : i64, alpha = 1.702000e+00 : f32, limit = 5.000000e+00 : f32}, decomposition = @tenstorrent.moe_gpt_decode.impl} : (tensor<4x1x1x128xbf16>, tensor<4x1x1x2xi64>, tensor<4x1x1x2xbf16>, tensor<1x1x2x8xi64>, tensor<1x1x2x8xi64>, tensor<12x1x8x4x160x128xbf16>, tensor<12x1x8x2x160x128xbf16>) -> tensor<2x1x4x128xbf16>
    return %0 : tensor<2x1x4x128xbf16>
  }

  func.func private @tenstorrent.moe_gpt_decode.impl(%arg0: tensor<4x1x1x128xbf16>, %arg1: tensor<4x1x1x2xi64>, %arg2: tensor<4x1x1x2xbf16>, %arg3: tensor<1x1x2x8xi64>, %arg4: tensor<1x1x2x8xi64>, %arg5: tensor<12x1x8x4x160x128xbf16>, %arg6: tensor<12x1x8x2x160x128xbf16>) -> tensor<2x1x4x128xbf16> {
    %0:3 = stablehlo.custom_call @tt.all_to_all_dispatch_metadata(%arg0, %arg1, %arg2, %arg3) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2"}} : (tensor<4x1x1x128xbf16>, tensor<4x1x1x2xi64>, tensor<4x1x1x2xbf16>, tensor<1x1x2x8xi64>) -> (tensor<1x8x1x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>)
    %1:5 = stablehlo.custom_call @tt.moe_gpt(%0#0, %0#1, %0#2, %arg4, %arg5, %arg6) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts = "8", num_experts_per_tok = "2"}} : (tensor<1x8x1x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x1x2x8xi64>, tensor<12x1x8x4x160x128xbf16>, tensor<12x1x8x2x160x128xbf16>) -> (tensor<1x1x2x8xi64>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x8x1x2xbf16>, tensor<8x1x8x128xbf16>)
    %2 = stablehlo.custom_call @tt.selective_reduce_combine(%1#4, %1#1, %1#2, %1#0) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "2", output_shard_dim = "2"}} : (tensor<8x1x8x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x1x2x8xi64>) -> tensor<2x1x4x128xbf16>
    return %2 : tensor<2x1x4x128xbf16>
  }
}
