// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @main(%arg0: tensor<1x64x512x512xf32>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x512x512xf32> {
    %0 = ttir.empty() : tensor<1x64x512x512xf32>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x64x512x512xf32>, tensor<64x64x3x3xf32>, tensor<1x64x512x512xf32>) -> tensor<1x64x512x512xf32>
    %2 = ttir.empty() : tensor<64x1x1xf32>
    %3 = "ttir.reshape"(%arg2, %2) <{shape = [64 : i32, 1 : i32, 1 : i32]}> : (tensor<64xf32>, tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
    %4 = ttir.empty() : tensor<1x64x512x512xf32>
    %5 = "ttir.broadcast"(%1, %4) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x64x512x512xf32>, tensor<1x64x512x512xf32>) -> tensor<1x64x512x512xf32>
    %6 = ttir.empty() : tensor<1x64x1x1xf32>
    %7 = "ttir.reshape"(%3, %6) <{shape = [1 : i32, 64 : i32, 1 : i32, 1 : i32]}> : (tensor<64x1x1xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %8 = ttir.empty() : tensor<1x64x512x512xf32>
    %9 = "ttir.broadcast"(%7, %8) <{broadcast_dimensions = array<i64: 1, 1, 512, 512>}> : (tensor<1x64x1x1xf32>, tensor<1x64x512x512xf32>) -> tensor<1x64x512x512xf32>
    %10 = ttir.empty() : tensor<1x64x512x512xf32>
    %11 = "ttir.add"(%5, %9, %10) : (tensor<1x64x512x512xf32>, tensor<1x64x512x512xf32>, tensor<1x64x512x512xf32>) -> tensor<1x64x512x512xf32>
    return %11 : tensor<1x64x512x512xf32>
  }
}

