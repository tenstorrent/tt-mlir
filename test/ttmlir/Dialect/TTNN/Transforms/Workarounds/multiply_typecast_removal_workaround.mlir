// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<64x32x1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<64x32x1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module @test_multiply_typecast_removal attributes {} {
  func.func public @test_multiply_typecast_removal_workaround(%arg0: tensor<512x512x1x1xbf16, #ttnn_layout_bf16>, %arg1: tensor<512x512x1x1xbf16, #ttnn_layout_bf16>) -> tensor<512x512x1x1xbf16, #ttnn_layout_bf16> {
    // CHECK-LABEL: func.func public @test_multiply_typecast_removal_workaround
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[MULTIPLY:[0-9]+]] = "ttnn.multiply"(%arg0, %arg1)
    // CHECK-SAME: <{dtype = #ttcore.supportedDataTypes<bf16>}>
    // CHECK-SAME: tensor<512x512x1x1xbf16
    // CHECK: return %[[MULTIPLY]]
    %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<512x512x1x1xbf16, #ttnn_layout_bf16>) -> tensor<512x512x1x1xf32, #ttnn_layout_f32>
    %1 = "ttnn.typecast"(%arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<512x512x1x1xbf16, #ttnn_layout_bf16>) -> tensor<512x512x1x1xf32, #ttnn_layout_f32>
    %2 = "ttnn.multiply"(%0, %1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<512x512x1x1xf32, #ttnn_layout_f32>, tensor<512x512x1x1xf32, #ttnn_layout_f32>) -> tensor<512x512x1x1xf32, #ttnn_layout_f32>
    %3 = "ttnn.typecast"(%2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<512x512x1x1xf32, #ttnn_layout_f32>) -> tensor<512x512x1x1xbf16, #ttnn_layout_bf16>
    return %3 : tensor<512x512x1x1xbf16, #ttnn_layout_bf16>
  }

  func.func public @test_multiply_no_workaround_no_typecast(%arg0: tensor<1x1x128x160xf32, #ttnn_layout_f32>, %arg1: tensor<1x1x128x160xf32, #ttnn_layout_f32>) -> tensor<1x1x128x160xf32, #ttnn_layout_f32> {
    // CHECK-LABEL: func.func public @test_multiply_no_workaround_no_typecast
    // CHECK: "ttnn.multiply"(%arg0, %arg1)
    // CHECK-NOT: "ttnn.typecast"
    %0 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1x128x160xf32, #ttnn_layout_f32>, tensor<1x1x128x160xf32, #ttnn_layout_f32>) -> tensor<1x1x128x160xf32, #ttnn_layout_f32>
    return %0 : tensor<1x1x128x160xf32, #ttnn_layout_f32>
  }

  func.func public @test_multiply_no_workaround_wrong_dtype(%arg0: tensor<2048x1024x1x1xf32, #ttnn_layout_f32>, %arg1: tensor<2048x1024x1x1xf32, #ttnn_layout_f32>) -> tensor<2048x1024x1x1xf32, #ttnn_layout_f32> {
    // CHECK-LABEL: func.func public @test_multiply_no_workaround_wrong_dtype
    // CHECK: "ttnn.multiply"(%arg0, %arg1)
    // CHECK-NOT: "ttnn.typecast"
    %0 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<2048x1024x1x1xf32, #ttnn_layout_f32>, tensor<2048x1024x1x1xf32, #ttnn_layout_f32>) -> tensor<2048x1024x1x1xf32, #ttnn_layout_f32>
    return %0 : tensor<2048x1024x1x1xf32, #ttnn_layout_f32>
  }

  func.func public @test_multiply_no_workaround_multiple_users(%arg0: tensor<512x512x1x1xbf16, #ttnn_layout_bf16>, %arg1: tensor<512x512x1x1xbf16, #ttnn_layout_bf16>) -> (tensor<512x512x1x1xbf16, #ttnn_layout_bf16>, tensor<512x512x1x1xf32, #ttnn_layout_f32>) {
    // CHECK-LABEL: func.func public @test_multiply_no_workaround_multiple_users
    // CHECK: %[[TYPECAST0:[0-9]+]] = "ttnn.typecast"(%arg0)
    // CHECK: %[[TYPECAST1:[0-9]+]] = "ttnn.typecast"(%arg1)
    // CHECK: %[[MULTIPLY:[0-9]+]] = "ttnn.multiply"(%[[TYPECAST0]], %[[TYPECAST1]])
    // CHECK: %[[TYPECAST2:[0-9]+]] = "ttnn.typecast"(%[[MULTIPLY]])
    // CHECK: return %[[TYPECAST2]], %[[MULTIPLY]]
    %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<512x512x1x1xbf16, #ttnn_layout_bf16>) -> tensor<512x512x1x1xf32, #ttnn_layout_f32>
    %1 = "ttnn.typecast"(%arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<512x512x1x1xbf16, #ttnn_layout_bf16>) -> tensor<512x512x1x1xf32, #ttnn_layout_f32>
    %2 = "ttnn.multiply"(%0, %1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<512x512x1x1xf32, #ttnn_layout_f32>, tensor<512x512x1x1xf32, #ttnn_layout_f32>) -> tensor<512x512x1x1xf32, #ttnn_layout_f32>
    %3 = "ttnn.typecast"(%2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<512x512x1x1xf32, #ttnn_layout_f32>) -> tensor<512x512x1x1xbf16, #ttnn_layout_bf16>
    return %3, %2 : tensor<512x512x1x1xbf16, #ttnn_layout_bf16>, tensor<512x512x1x1xf32, #ttnn_layout_f32>
  }

  func.func public @test_multiply_skips_when_decomposition_applies(%arg0: tensor<2048x1024x1x1xbf16, #ttnn_layout_bf16>, %arg1: tensor<2048x1024x1x1xbf16, #ttnn_layout_bf16>) -> tensor<2048x1024x1x1xbf16, #ttnn_layout_bf16> {
    // CHECK-LABEL: func.func public @test_multiply_skips_when_decomposition_applies
    // This test ensures typecast removal doesn't interfere with the decomposition workaround
    // The decomposition pattern should add permutes for large tensors
    // CHECK: "ttnn.permute"
    %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<2048x1024x1x1xbf16, #ttnn_layout_bf16>) -> tensor<2048x1024x1x1xf32, #ttnn_layout_f32>
    %1 = "ttnn.typecast"(%arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<2048x1024x1x1xbf16, #ttnn_layout_bf16>) -> tensor<2048x1024x1x1xf32, #ttnn_layout_f32>
    %2 = "ttnn.multiply"(%0, %1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<2048x1024x1x1xf32, #ttnn_layout_f32>, tensor<2048x1024x1x1xf32, #ttnn_layout_f32>) -> tensor<2048x1024x1x1xf32, #ttnn_layout_f32>
    %3 = "ttnn.typecast"(%2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<2048x1024x1x1xf32, #ttnn_layout_f32>) -> tensor<2048x1024x1x1xbf16, #ttnn_layout_bf16>
    return %3 : tensor<2048x1024x1x1xbf16, #ttnn_layout_bf16>
  }
}
