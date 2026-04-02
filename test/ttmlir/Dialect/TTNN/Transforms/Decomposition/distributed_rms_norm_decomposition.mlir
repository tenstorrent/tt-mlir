// RUN: ttmlir-opt --ttcore-register-device --ttnn-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1, d2 * 128 + d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_weight = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_supported = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Test: Non-(1,1,32,M) shape decomposes into primitive ops.
module @test_distributed_rms_norm_decomposition attributes {} {
  func.func public @test_decompose_non_supported_shape(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_non_supported_shape
    // Pre all-gather: square and local mean
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.mean"
    // All-gather
    // CHECK: "ttnn.all_gather"
    // Post all-gather: global mean, add epsilon, rsqrt, normalize, apply weight
    // CHECK: "ttnn.mean"
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.rsqrt"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK-NOT: "ttnn.distributed_rms_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }

  func.func public @test_no_decompose_supported_shape(
      %arg0: tensor<1x1x32x128xbf16, #ttnn_layout_supported>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x1x32x128xbf16, #ttnn_layout_supported> {
    // CHECK-LABEL: func.func public @test_no_decompose_supported_shape
    // The (1,1,32,M) shape must NOT be decomposed — op survives for the fused kernel.
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-NOT: "ttnn.rsqrt"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (tensor<1x1x32x128xbf16, #ttnn_layout_supported>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_supported>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_supported>
  }

  func.func public @test_decompose_with_residual(
      %arg0: tensor<1x1x64x128xbf16, #ttnn_layout>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>,
      %arg2: tensor<1x1x64x128xbf16, #ttnn_layout>) -> tensor<1x1x64x128xbf16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_decompose_with_residual
    // With residual: first op must be add(input, residual).
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.mean"
    // CHECK: "ttnn.all_gather"
    // CHECK-NOT: "ttnn.distributed_rms_norm"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %arg2, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>}> : (tensor<1x1x64x128xbf16, #ttnn_layout>, tensor<128xbf16, #ttnn_layout_weight>, tensor<1x1x64x128xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x64x128xbf16, #ttnn_layout>
    return %1 : tensor<1x1x64x128xbf16, #ttnn_layout>
  }
}
