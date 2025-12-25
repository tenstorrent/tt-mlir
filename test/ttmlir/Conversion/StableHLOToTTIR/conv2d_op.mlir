// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_convolution attributes {} {
  func.func public @test_convolution(%arg0: tensor<1x128x128x32xf32>, %arg1: tensor<64x32x3x3xf32>) -> tensor<1x128x128x64xf32> {
    // CHECK: = "ttir.conv2d"
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
    // CHECK-COUNT-2: = "ttir.reshape"
    // CHECK: = "ttir.conv2d"
    // CHECK: = "ttir.reshape"
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
    // CHECK: [[CONV:%[0-9]+]] = "ttir.conv2d"
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0>
    // CHECK: [[SLICE:%[0-9]+]] = "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 1 : i32, 512 : i32, 1088 : i32]
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
