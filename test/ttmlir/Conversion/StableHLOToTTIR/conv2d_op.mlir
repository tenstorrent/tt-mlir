// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_convolution attributes {} {
  func.func public @test_convolution(%arg0: tensor<1x128x128x32xf32>, %arg1: tensor<64x32x3x3xf32>) -> tensor<1x128x128x64xf32> {
    // CHECK: = tensor.empty
    // CHECK: = "ttir.convolution"
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f],
      window = {
        stride = [1, 1],
        pad = [[1, 1], [1, 1]],
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64,
        precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
      } : (tensor<1x128x128x32xf32>, tensor<64x32x3x3xf32>) -> tensor<1x128x128x64xf32>
    return %0 : tensor<1x128x128x64xf32>
  }

  // Tests 1d convolution that gets translated to 2d.
  func.func @test_convolution_1d(%arg0: tensor<1x256x512xf32>, %arg1: tensor<1024x256x1xf32>) -> tensor<1x1024x512xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[TENSOR_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
    // CHECK: %1 = "ttir.convolution"(%arg0, %arg1, [[VAL0]])
    // CHECK: batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2
    // CHECK: padding = array<i64: 0, 0>, weight_dilation = array<i64: 1>, window_reversal = array<i1: false>, window_strides = array<i64: 1>
    // CHECK: : (tensor<1x256x512xf32>, tensor<1024x256x1xf32>, [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
      window = {
        stride = [1],
        pad = [[0, 0]],
        rhs_dilate = [1]
      } {
        batch_group_count = 1 : i64,
        feature_group_count = 1 : i64
      } : (tensor<1x256x512xf32>, tensor<1024x256x1xf32>) -> tensor<1x1024x512xf32>
    return %0 : tensor<1x1024x512xf32>
  }
}
