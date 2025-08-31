// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<64x32x1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_small = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x1x4x5x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module @test_multiply_decomposition attributes {} {
  func.func public @test_multiply_decomposition_workaround(%arg0: tensor<2048x1024x1x1xf32, #ttnn_layout>, %arg1: tensor<2048x1024x1x1xf32, #ttnn_layout>) -> tensor<2048x1024x1x1xf32, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_multiply_decomposition_workaround
    // Create some operations to produce the tensors (since the workaround now checks for non-null defining ops)
    %0 = "ttnn.relu"(%arg0) : (tensor<2048x1024x1x1xf32, #ttnn_layout>) -> tensor<2048x1024x1x1xf32, #ttnn_layout>
    %1 = "ttnn.relu"(%arg1) : (tensor<2048x1024x1x1xf32, #ttnn_layout>) -> tensor<2048x1024x1x1xf32, #ttnn_layout>
    // CHECK: %[[LHS_RELU:[0-9]+]] = "ttnn.relu"(%arg0)
    // CHECK: %[[LHS_PERMUTE:[0-9]+]] = "ttnn.permute"(%[[LHS_RELU]])
    // CHECK-SAME: permutation = array<i64: 2, 3, 0, 1>
    // CHECK: %[[RHS_RELU:[0-9]+]] = "ttnn.relu"(%arg1)
    // CHECK: %[[RHS_PERMUTE:[0-9]+]] = "ttnn.permute"(%[[RHS_RELU]])
    // CHECK-SAME: permutation = array<i64: 2, 3, 0, 1>
    // CHECK: %[[MULTIPLY:[0-9]+]] = "ttnn.multiply"(%[[LHS_PERMUTE]], %[[RHS_PERMUTE]])
    // CHECK: %[[OUTPUT_PERMUTE:[0-9]+]] = "ttnn.permute"(%[[MULTIPLY]])
    // CHECK-SAME: permutation = array<i64: 2, 3, 0, 1>
    // CHECK: return %[[OUTPUT_PERMUTE]]
    %2 = "ttnn.multiply"(%0, %1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<2048x1024x1x1xf32, #ttnn_layout>, tensor<2048x1024x1x1xf32, #ttnn_layout>) -> tensor<2048x1024x1x1xf32, #ttnn_layout>
    return %2 : tensor<2048x1024x1x1xf32, #ttnn_layout>
  }

  func.func public @test_multiply_no_workaround(%arg0: tensor<1x1x128x160xf32, #ttnn_layout_small>, %arg1: tensor<1x1x128x160xf32, #ttnn_layout_small>) -> tensor<1x1x128x160xf32, #ttnn_layout_small> {
    // CHECK-LABEL: func.func public @test_multiply_no_workaround
    // CHECK: "ttnn.multiply"(%arg0, %arg1)
    // CHECK-NOT: "ttnn.permute"
    %0 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1x128x160xf32, #ttnn_layout_small>, tensor<1x1x128x160xf32, #ttnn_layout_small>) -> tensor<1x1x128x160xf32, #ttnn_layout_small>
    return %0 : tensor<1x1x128x160xf32, #ttnn_layout_small>
  }
}
