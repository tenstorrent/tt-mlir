// REQUIRES: stablehlo
// RUN: ttmlir-opt --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// The moe_gpt_decode composite uses a 4D layout at the composite boundary
// ([B, 1, S, H] for hidden/topk and [K, S, B, H] for the combined result) so
// it can directly bridge the surrounding MoE custom calls
// (tt.all_to_all_dispatch_metadata / tt.moe_gpt / tt.selective_reduce_combine)
// without any reshape at the boundary.
module @gpt_oss_decode_composite_tests attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  // CHECK-LABEL: func.func @moe_gpt_decode
  // CHECK: %[[RESULT:.*]] = "ttir.moe_gpt_decode"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8)
  // CHECK-SAME: alpha =
  // CHECK-SAME: cluster_axis = 1 : i64
  // CHECK-SAME: intermediate_size = 128 : i64
  // CHECK-SAME: limit =
  // CHECK-SAME: num_devices = 2 : i64
  // CHECK-SAME: num_experts = 8 : i64
  // CHECK-SAME: num_experts_per_tok = 2 : i64
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0>
  func.func @moe_gpt_decode(%arg0: tensor<4x1x1x128xbf16>, %arg1: tensor<4x1x1x2xi64>, %arg2: tensor<4x1x1x2xbf16>, %arg3: tensor<1x1x2x8xi64>, %arg4: tensor<1x1x2x8xi64>, %arg5: tensor<8x128x256xbf16>, %arg6: tensor<8x256xbf16>, %arg7: tensor<8x128x128xbf16>, %arg8: tensor<8x128xbf16>) -> tensor<2x1x4x128xbf16> {
    // CHECK: return %[[RESULT]] : tensor<2x1x4x128xbf16>
    %0 = stablehlo.composite "tenstorrent.moe_gpt_decode" %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 {composite_attributes = {num_devices = 2 : i64, cluster_axis = 1 : i64, num_experts = 8 : i64, num_experts_per_tok = 2 : i64, intermediate_size = 128 : i64, alpha = 1.702000e+00 : f32, limit = 5.000000e+00 : f32}, decomposition = @tenstorrent.moe_gpt_decode.impl} : (tensor<4x1x1x128xbf16>, tensor<4x1x1x2xi64>, tensor<4x1x1x2xbf16>, tensor<1x1x2x8xi64>, tensor<1x1x2x8xi64>, tensor<8x128x256xbf16>, tensor<8x256xbf16>, tensor<8x128x128xbf16>, tensor<8x128xbf16>) -> tensor<2x1x4x128xbf16>
    return %0 : tensor<2x1x4x128xbf16>
  }

  func.func private @tenstorrent.moe_gpt_decode.impl(%arg0: tensor<4x1x1x128xbf16>, %arg1: tensor<4x1x1x2xi64>, %arg2: tensor<4x1x1x2xbf16>, %arg3: tensor<1x1x2x8xi64>, %arg4: tensor<1x1x2x8xi64>, %arg5: tensor<8x128x256xbf16>, %arg6: tensor<8x256xbf16>, %arg7: tensor<8x128x128xbf16>, %arg8: tensor<8x128xbf16>) -> tensor<2x1x4x128xbf16> {
    %0:3 = stablehlo.custom_call @tt.all_to_all_dispatch_metadata(%arg0, %arg1, %arg2, %arg3) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2"}} : (tensor<4x1x1x128xbf16>, tensor<4x1x1x2xi64>, tensor<4x1x1x2xbf16>, tensor<1x1x2x8xi64>) -> (tensor<1x8x1x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>)
    %1:5 = stablehlo.custom_call @tt.moe_gpt(%0#0, %0#1, %0#2, %arg4, %arg5, %arg6, %arg7, %arg8) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "2"}} : (tensor<1x8x1x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x1x2x8xi64>, tensor<8x128x256xbf16>, tensor<8x256xbf16>, tensor<8x128x128xbf16>, tensor<8x128xbf16>) -> (tensor<1x1x2x8xi64>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x8x1x2xbf16>, tensor<8x1x8x128xbf16>)
    %2 = stablehlo.custom_call @tt.selective_reduce_combine(%1#4, %1#1, %1#2, %1#0) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "2", output_shard_dim = "2"}} : (tensor<8x1x8x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x1x2x8xi64>) -> tensor<2x1x4x128xbf16>
    return %2 : tensor<2x1x4x128xbf16>
  }

  // CHECK-LABEL: func.func @moe_gpt_decode_fused
  // CHECK: %[[RESULT:.*]] = "ttir.moe_gpt_decode"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10)
  // CHECK-SAME: has_fused_weights
  // CHECK-NOT: has_fused_weights
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>
  // Verifies that:
  //   * both fused_w0_w1 / fused_w2 survive the composite -> ttir.moe_gpt_decode
  //     legalization as operands 9 and 10, and
  //   * the `has_fused_weights = true` composite attribute is stripped (it is
  //     implied by the presence of the operands via operandSegmentSizes and
  //     the op does not declare it).
  // The fused weights follow tt-metal's `create_fused_moe_gpt_config` layout:
  //   w0_w1: [C_dram, L, E_local, 4, K+32, 4*TILE]
  //   w2:    [C_dram, L, E_local, 2, N+32, 4*TILE]
  func.func @moe_gpt_decode_fused(%arg0: tensor<4x1x1x128xbf16>, %arg1: tensor<4x1x1x2xi64>, %arg2: tensor<4x1x1x2xbf16>, %arg3: tensor<1x1x2x8xi64>, %arg4: tensor<1x1x2x8xi64>, %arg5: tensor<8x128x256xbf16>, %arg6: tensor<8x256xbf16>, %arg7: tensor<8x128x128xbf16>, %arg8: tensor<8x128xbf16>, %arg9: tensor<12x1x8x4x160x128xbf16>, %arg10: tensor<12x1x8x2x160x128xbf16>) -> tensor<2x1x4x128xbf16> {
    // CHECK: return %[[RESULT]] : tensor<2x1x4x128xbf16>
    %0 = stablehlo.composite "tenstorrent.moe_gpt_decode" %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 {composite_attributes = {num_devices = 2 : i64, cluster_axis = 1 : i64, num_experts = 8 : i64, num_experts_per_tok = 2 : i64, intermediate_size = 128 : i64, alpha = 1.702000e+00 : f32, limit = 5.000000e+00 : f32, has_fused_weights = true}, decomposition = @tenstorrent.moe_gpt_decode_fused.impl} : (tensor<4x1x1x128xbf16>, tensor<4x1x1x2xi64>, tensor<4x1x1x2xbf16>, tensor<1x1x2x8xi64>, tensor<1x1x2x8xi64>, tensor<8x128x256xbf16>, tensor<8x256xbf16>, tensor<8x128x128xbf16>, tensor<8x128xbf16>, tensor<12x1x8x4x160x128xbf16>, tensor<12x1x8x2x160x128xbf16>) -> tensor<2x1x4x128xbf16>
    return %0 : tensor<2x1x4x128xbf16>
  }

  func.func private @tenstorrent.moe_gpt_decode_fused.impl(%arg0: tensor<4x1x1x128xbf16>, %arg1: tensor<4x1x1x2xi64>, %arg2: tensor<4x1x1x2xbf16>, %arg3: tensor<1x1x2x8xi64>, %arg4: tensor<1x1x2x8xi64>, %arg5: tensor<8x128x256xbf16>, %arg6: tensor<8x256xbf16>, %arg7: tensor<8x128x128xbf16>, %arg8: tensor<8x128xbf16>, %arg9: tensor<12x1x8x4x160x128xbf16>, %arg10: tensor<12x1x8x2x160x128xbf16>) -> tensor<2x1x4x128xbf16> {
    %0:3 = stablehlo.custom_call @tt.all_to_all_dispatch_metadata(%arg0, %arg1, %arg2, %arg3) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2"}} : (tensor<4x1x1x128xbf16>, tensor<4x1x1x2xi64>, tensor<4x1x1x2xbf16>, tensor<1x1x2x8xi64>) -> (tensor<1x8x1x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>)
    %1:5 = stablehlo.custom_call @tt.moe_gpt(%0#0, %0#1, %0#2, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "2", has_fused_weights = "true"}} : (tensor<1x8x1x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x1x2x8xi64>, tensor<8x128x256xbf16>, tensor<8x256xbf16>, tensor<8x128x128xbf16>, tensor<8x128xbf16>, tensor<12x1x8x4x160x128xbf16>, tensor<12x1x8x2x160x128xbf16>) -> (tensor<1x1x2x8xi64>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x8x1x2xbf16>, tensor<8x1x8x128xbf16>)
    %2 = stablehlo.custom_call @tt.selective_reduce_combine(%1#4, %1#1, %1#2, %1#0) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "2", num_experts_per_tok = "2", output_shard_dim = "2"}} : (tensor<8x1x8x128xbf16>, tensor<1x8x1x2xi64>, tensor<1x8x1x2xbf16>, tensor<1x1x2x8xi64>) -> tensor<2x1x4x128xbf16>
    return %2 : tensor<2x1x4x128xbf16>
  }
}
