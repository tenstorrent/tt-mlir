// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @conv_transpose attributes {} {
  func.func @test_conv_transpose2d(%arg0: tensor<1x256x32x32xf32>, %arg1: tensor<256x128x2x2xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x64x64xf32> {
    // CHECK-LABEL: @test_conv_transpose2d
    %0 = stablehlo.transpose %arg1, dims = [2, 3, 1, 0] : (tensor<256x128x2x2xf32>) -> tensor<2x2x128x256xf32>
    %1 = stablehlo.reverse %0, dims = [0, 1] : tensor<2x2x128x256xf32>
    // CHECK: %{{[0-9]+}} = "ttir.convolution"(%arg0, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: batch_group_count = 1 : i64,
    // CHECK-SAME: convolution_layout = #ttir<convolution_layout input_batch = 0,
    // CHECK-SAME: input_feature = 1,
    // CHECK-SAME: input_spatial_dimensions = 2x3,
    // CHECK-SAME: kernel_output_feature = 2,
    // CHECK-SAME: kernel_input_feature = 3,
    // CHECK-SAME: kernel_spatial_dimensions = 0x1,
    // CHECK-SAME: output_batch = 0,
    // CHECK-SAME: output_feature = 1,
    // CHECK-SAME: output_spatial_dimensions = 2x3>,
    // CHECK-SAME: feature_group_count = 1 : i64,
    // CHECK-SAME: input_dilation = array<i64: 2, 2>,
    // CHECK-SAME: padding = array<i64: 1, 1, 1, 1>,
    // CHECK-SAME: weight_dilation = array<i64: 1, 1>,
    // CHECK-SAME: window_reversal = array<i1: false, false>,
    // CHECK-SAME: window_strides = array<i64: 1, 1>
    // CHECK-SAME: (tensor<1x256x32x32xf32>, tensor<2x2x128x256xf32>, tensor<1x128x64x64xf32>)
    // CHECK-SAME: -> tensor<1x128x64x64xf32>
    %2 = stablehlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x256x32x32xf32>, tensor<2x2x128x256xf32>) -> tensor<1x128x64x64xf32>
    %3 = stablehlo.reshape %arg2 : (tensor<128xf32>) -> tensor<128x1x1xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %5 = stablehlo.broadcast_in_dim %3, dims = [1, 2, 3] : (tensor<128x1x1xf32>) -> tensor<1x128x64x64xf32>
    %6 = stablehlo.add %4, %5 : tensor<1x128x64x64xf32>
    return %6 : tensor<1x128x64x64xf32>
  }
}
