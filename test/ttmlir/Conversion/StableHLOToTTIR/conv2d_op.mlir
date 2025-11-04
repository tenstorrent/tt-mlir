// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_convolution attributes {} {
  func.func public @test_convolution(%arg0: tensor<1x128x128x32xf32>, %arg1: tensor<64x32x3x3xf32>) -> tensor<1x128x128x64xf32> {
    // CHECK: = ttir.empty
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
    // CHECK: [[VAL0:%[0-9]+]] = ttir.empty() : [[TENSOR_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
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

  // Tests convolution with negative padding.
  func.func @test_convolution_negative_padding(%arg0: tensor<1x14x14x512xf32>, %arg1: tensor<1x7x7x1088xf32>) -> tensor<1x1x512x1088xf32> {
    // CHECK: [[CONV_OUT:%[0-9]+]] = ttir.empty() : tensor<2x2x512x1088xf32>
    // CHECK: [[CONV:%[0-9]+]] = "ttir.convolution"(%arg0, %arg1, [[CONV_OUT]])
    // CHECK-SAME: batch_group_count = 1 : i64
    // CHECK-SAME: convolution_layout = #ttir<convolution_layout input_batch = 3, input_feature = 0, input_spatial_dimensions = 1x2, kernel_output_feature = 3, kernel_input_feature = 0, kernel_spatial_dimensions = 1x2, output_batch = 2, output_feature = 3, output_spatial_dimensions = 0x1>
    // CHECK-SAME: padding = array<i64: 0, 0, 0, 0>
    // CHECK-SAME: weight_dilation = array<i64: 2, 2>
    // CHECK-SAME: : (tensor<1x14x14x512xf32>, tensor<1x7x7x1088xf32>, tensor<2x2x512x1088xf32>) -> tensor<2x2x512x1088xf32>
    // CHECK: [[SLICE_OUT:%[0-9]+]] = ttir.empty() : tensor<1x1x512x1088xf32>
    // CHECK: [[SLICE:%[0-9]+]] = "ttir.slice_static"([[CONV]], [[SLICE_OUT]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 1 : i32, 512 : i32, 1088 : i32]
    // CHECK-SAME: : (tensor<2x2x512x1088xf32>, tensor<1x1x512x1088xf32>) -> tensor<1x1x512x1088xf32>
    %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f],
    window = {
      pad = [[0, -1], [0, -1]],
      rhs_dilate = [2, 2]
    } {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
    } : (tensor<1x14x14x512xf32>, tensor<1x7x7x1088xf32>) -> tensor<1x1x512x1088xf32>
    return %0 : tensor<1x1x512x1088xf32>
  }
}
