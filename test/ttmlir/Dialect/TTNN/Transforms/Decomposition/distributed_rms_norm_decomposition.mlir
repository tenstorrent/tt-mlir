// RUN: ttmlir-opt --ttcore-register-device --ttnn-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1, d2 * 128 + d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_weight = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_supported = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_supported_rank3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_supported_rank3_h64 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_unsupported_leading = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2 + d1, d2 * 128 + d3), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Test: Non-(1,1,32,M) shape decomposes into primitive ops.
module @test_distributed_rms_norm_decomposition attributes {} {
  func.func public @test_decompose_non_supported_shape(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_non_supported_shape
    // Pre all-gather:
    // CHECK: "ttnn.rms_norm_pre_all_gather"
    // All-gather
    // CHECK: "ttnn.all_gather"
    // Post all-gather: global mean, add epsilon, rsqrt, normalize, apply weight
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.mean"
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.rsqrt"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK-NOT: "ttnn.distributed_rms_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }

  func.func public @test_no_decompose_supported_shape(
      %arg0: tensor<1x1x32x128xbf16, #ttnn_layout_supported>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x1x32x128xbf16, #ttnn_layout_supported> {
    // CHECK-LABEL: func.func public @test_no_decompose_supported_shape
    // The (1,1,32,M) shape stays on the fused op, but the 1D weight (N,)
    // must still be reshaped to 2D (N/32, 32) so the fused kernel can run.
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 32 : i32]
    // CHECK-SAME: -> tensor<4x32
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-SAME: tensor<1x1x32x128
    // CHECK-NOT: "ttnn.rsqrt"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>}> : (tensor<1x1x32x128xbf16, #ttnn_layout_supported>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_supported>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_supported>
  }

  func.func public @test_reshape_rank3_to_canonical_shape(
      %arg0: tensor<1x32x128xbf16, #ttnn_layout_supported_rank3>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x32x128xbf16, #ttnn_layout_supported_rank3> {
    // CHECK-LABEL: func.func public @test_reshape_rank3_to_canonical_shape
    // Rank-3 (1, 32, M) is eligible for the fused kernel but not yet canonical.
    // The decomposition pass wraps it: reshape weight to 2D, reshape input to
    // (1,1,32,M), forward to the fused op, then reshape the result back to
    // (1,32,M).
    // Weight is reshaped from 1D (128) to 2D (4, 32).
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 32 : i32]
    // CHECK-SAME: -> tensor<4x32
    // Input is reshaped to (1, 1, 32, 128).
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 32 : i32, 128 : i32]
    // CHECK-SAME: -> tensor<1x1x32x128
    // The fused op is kept on the canonical shape.
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-SAME: tensor<1x1x32x128
    // CHECK-NOT: "ttnn.rsqrt"
    // Result is reshaped back to (1, 32, 128).
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 32 : i32, 128 : i32]
    // CHECK-SAME: -> tensor<1x32x128
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>}> : (tensor<1x32x128xbf16, #ttnn_layout_supported_rank3>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x32x128xbf16, #ttnn_layout_supported_rank3>
    return %1 : tensor<1x32x128xbf16, #ttnn_layout_supported_rank3>
  }

  func.func public @test_decompose_unsupported_leading_dim(
      %arg0: tensor<1x2x32x128xbf16, #ttnn_layout_unsupported_leading>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x2x32x128xbf16, #ttnn_layout_unsupported_leading> {
    // CHECK-LABEL: func.func public @test_decompose_unsupported_leading_dim
    // Leading dim != 1 means the input cannot be reshaped to (1,1,32,M)
    // without data movement, so the op must be decomposed here.
    // CHECK: "ttnn.rms_norm_pre_all_gather"
    // CHECK: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.distributed_rms_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>}> : (tensor<1x2x32x128xbf16, #ttnn_layout_unsupported_leading>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x2x32x128xbf16, #ttnn_layout_unsupported_leading>
    return %1 : tensor<1x2x32x128xbf16, #ttnn_layout_unsupported_leading>
  }

  func.func public @test_decompose_with_residual(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>,
      %arg2: tensor<1x1x64x128xbf16, #ttnn_layout>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_with_residual
    // With residual: first op must be add(input, residual).
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.rms_norm_pre_all_gather"
    // CHECK: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.distributed_rms_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %arg2, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, tensor<128xbf16, #ttnn_layout_weight>, tensor<1x1x64x128xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }

  func.func public @test_reshape_rank3_non_fused_to_pre_all_gather_shape(
      %arg0: tensor<1x64x128xbf16, #ttnn_layout_supported_rank3_h64>) -> tensor<1x64x128xbf16, #ttnn_layout_supported_rank3_h64> {
    // CHECK-LABEL: func.func public @test_reshape_rank3_non_fused_to_pre_all_gather_shape
    // CHECK-NOT: "ttnn.distributed_rms_norm"
    // Rank-3 input with no weight is not eligible for the fused kernel, so it will lower through
    // rms_norm_pre_all_gather. That runtime expects rank-4 input, so the pass
    // first rewrites distributed_rms_norm at shape (1,1,64,M), then reshapes the
    // result back to (1,64,M).
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 64 : i32, 128 : i32]
    // CHECK-SAME: -> tensor<1x1x64x128
    // CHECK: "ttnn.rms_norm_pre_all_gather"
    // CHECK-SAME: tensor<1x1x64x128
    // CHECK: "ttnn.all_gather"
    // Result is reshaped back to (1, 64, 128).
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<1x1x64x128
    // CHECK-SAME: -> tensor<1x64x128
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 1>}> : (tensor<1x64x128xbf16, #ttnn_layout_supported_rank3_h64>, !ttnn.device) -> tensor<1x64x128xbf16, #ttnn_layout_supported_rank3_h64>
    return %1 : tensor<1x64x128xbf16, #ttnn_layout_supported_rank3_h64>
  }

  // Cross-device mean of per-device sum(x^2) must be scaled by 1/N
  // (N=128 -> 7.812500e-03) before adding epsilon.
  func.func public @test_pre_all_gather_divides_stats_by_hidden_size(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_pre_all_gather_divides_stats_by_hidden_size
    // CHECK: %[[MEAN:[0-9]+]] = "ttnn.mean"
    // CHECK: %[[INVH:[0-9]+]] = "ttnn.full"(%{{[0-9]+}}) <{fill_value = 7.812500e-03 : f32
    // CHECK: %[[EX2:[0-9]+]] = "ttnn.multiply"(%[[MEAN]], %[[INVH]])
    // CHECK: "ttnn.add"(%[[EX2]],
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }
}
