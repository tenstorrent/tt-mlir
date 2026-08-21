// RUN: ttmlir-opt --ttcore-register-device --ttnn-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
// Layout for input/result: (1, 1, 64, 128)
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1, d2 * 128 + d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Layout for weight/bias: (128,)
#ttnn_layout_weight = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Layout for rank-3 input/result: (1, 64, 128) — Wan AdaLN is 1xLxD.
#ttnn_layout_rank3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Test: Basic decomposition (no weight, bias, or residual).
// Verifies that the three dedicated TTNN ops are emitted and the intermediate
// ttnn.distributed_layer_norm op is fully consumed.
module @test_distributed_layer_norm_decomposition attributes {} {
  func.func public @test_decompose_basic(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_basic
    // Pre all-gather phase. Metal Welford requires fp32_dest_acc_en for f32
    // inputs; the decomp always emits the high-precision compute config.
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    // CHECK-SAME: fp32_dest_acc_en = true
    // All-gather statistics across devices
    // CHECK: "ttnn.all_gather"
    // Post all-gather: normalize using gathered stats
    // CHECK: "ttnn.layer_norm_post_all_gather"
    // CHECK-SAME: fp32_dest_acc_en = true
    // No explicit residual add expected
    // CHECK-NOT: "ttnn.add"
    // Fully decomposed — no intermediate distributed_layer_norm op remains
    // CHECK-NOT: "ttnn.distributed_layer_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_layer_norm"(%arg0, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }

  func.func public @test_decompose_with_weight_and_bias(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>,
      %arg2: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_with_weight_and_bias
    // Pre all-gather phase
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    // All-gather statistics across devices
    // CHECK: "ttnn.all_gather"
    // Post all-gather should receive weight and bias
    // CHECK: "ttnn.layer_norm_post_all_gather"
    // CHECK-NOT: "ttnn.distributed_layer_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_layer_norm"(%arg0, %arg1, %arg2, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, tensor<128xbf16, #ttnn_layout_weight>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }

  func.func public @test_decompose_with_residual(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>,
      %arg1: tensor<1x1x64x128xbf16, #ttnn_layout>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_with_residual
    // With residual: first op must be add(input, residual)
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.layer_norm_post_all_gather"
    // CHECK-NOT: "ttnn.distributed_layer_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_layer_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 1, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, tensor<1x1x64x128xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }

  func.func public @test_decompose_full(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>,
      %arg2: tensor<128xbf16, #ttnn_layout_weight>,
      %arg3: tensor<1x1x64x128xbf16, #ttnn_layout>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_full
    // Full: weight + bias + residual
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.layer_norm_post_all_gather"
    // CHECK-NOT: "ttnn.distributed_layer_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_layer_norm"(%arg0, %arg1, %arg2, %arg3, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, tensor<128xbf16, #ttnn_layout_weight>, tensor<128xbf16, #ttnn_layout_weight>, tensor<1x1x64x128xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }

  func.func public @test_reshape_rank3_to_pre_all_gather_shape(
      %arg0: tensor<1x64x128xbf16, #ttnn_layout_rank3>) -> tensor<1x64x128xbf16, #ttnn_layout_rank3> {
    // CHECK-LABEL: func.func public @test_reshape_rank3_to_pre_all_gather_shape
    // CHECK-NOT: "ttnn.distributed_layer_norm"
    // Rank-3 input is padded to rank 4 for layer_norm_pre_all_gather, then the
    // result is reshaped back to (1, 64, 128).
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 64 : i32, 128 : i32]
    // CHECK-SAME: -> tensor<1x1x64x128
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    // CHECK-SAME: tensor<1x1x64x128
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.layer_norm_post_all_gather"
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 64 : i32, 128 : i32]}> : (tensor<1x1x64x128
    // CHECK-SAME: -> tensor<1x64x128
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_layer_norm"(%arg0, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0, 1>}> : (tensor<1x64x128xbf16, #ttnn_layout_rank3>, !ttnn.device) -> tensor<1x64x128xbf16, #ttnn_layout_rank3>
    return %1 : tensor<1x64x128xbf16, #ttnn_layout_rank3>
  }

  func.func public @test_reshape_rank3_with_weight_and_bias(
      %arg0: tensor<1x64x128xbf16, #ttnn_layout_rank3>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>,
      %arg2: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x64x128xbf16, #ttnn_layout_rank3> {
    // CHECK-LABEL: func.func public @test_reshape_rank3_with_weight_and_bias
    // CHECK-NOT: "ttnn.distributed_layer_norm"
    // Weight/bias stay 1D; only the activation is rank-promoted.
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 64 : i32, 128 : i32]
    // CHECK: "ttnn.layer_norm_pre_all_gather"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.layer_norm_post_all_gather"
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 64 : i32, 128 : i32]}>
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_layer_norm"(%arg0, %arg1, %arg2, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>}> : (tensor<1x64x128xbf16, #ttnn_layout_rank3>, tensor<128xbf16, #ttnn_layout_weight>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x64x128xbf16, #ttnn_layout_rank3>
    return %1 : tensor<1x64x128xbf16, #ttnn_layout_rank3>
  }
}
