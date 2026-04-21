// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_supported = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_supported_rank3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_weight = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Test: (1,1,32,M) shape must NOT be decomposed by workarounds — the fused kernel handles it.
module @test_distributed_rms_norm_workaround attributes {} {
  func.func public @test_no_decompose_supported_shape(
      %arg0: tensor<1x1x32x128xbf16, #ttnn_layout_supported>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x1x32x128xbf16, #ttnn_layout_supported> {
    // CHECK-LABEL: func.func public @test_no_decompose_supported_shape
    // The (1,1,32,M) shape must survive as distributed_rms_norm.
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-NOT: "ttnn.rsqrt"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (tensor<1x1x32x128xbf16, #ttnn_layout_supported>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x1x32x128xbf16, #ttnn_layout_supported>
    return %1 : tensor<1x1x32x128xbf16, #ttnn_layout_supported>
  }

  // Test: rank-3 (1, 32, M) input gets reshaped to (1, 1, 32, M) before the
  // op and reshaped back to (1, 32, M) after.
  func.func public @test_reshape_rank3_to_canonical_shape(
      %arg0: tensor<1x32x128xbf16, #ttnn_layout_supported_rank3>,
      %arg1: tensor<128xbf16, #ttnn_layout_weight>) -> tensor<1x32x128xbf16, #ttnn_layout_supported_rank3> {
    // CHECK-LABEL: func.func public @test_reshape_rank3_to_canonical_shape
    // Input is reshaped to (1, 1, 32, 128).
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 32 : i32, 128 : i32]
    // CHECK-SAME: -> tensor<1x1x32x128
    // The fused op survives on the canonical shape.
    // CHECK: "ttnn.distributed_rms_norm"
    // CHECK-SAME: tensor<1x1x32x128
    // Result is reshaped back to (1, 32, 128).
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 32 : i32, 128 : i32]
    // CHECK-SAME: -> tensor<1x32x128
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.distributed_rms_norm"(%arg0, %arg1, %0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (tensor<1x32x128xbf16, #ttnn_layout_supported_rank3>, tensor<128xbf16, #ttnn_layout_weight>, !ttnn.device) -> tensor<1x32x128xbf16, #ttnn_layout_supported_rank3>
    return %1 : tensor<1x32x128xbf16, #ttnn_layout_supported_rank3>
  }
}
