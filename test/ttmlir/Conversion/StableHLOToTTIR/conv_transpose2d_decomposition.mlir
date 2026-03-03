// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module @test_conv_transpose {
  func.func @test_conv_transpose2d(%arg0: tensor<1x256x32x32xf32>, %arg1: tensor<256x128x2x2xf32>) -> tensor<1x128x64x64xf32> {
    // CHECK-LABEL: @test_conv_transpose2d
    // CHECK-NOT: ttir.reverse
    // CHECK: %[[CONV_T:[0-9]+]] = "ttir.conv_transpose2d"
    // CHECK-SAME: dilation = array<i32: 1, 1>,
    // CHECK-SAME: groups = 1 : i32,
    // CHECK-SAME: output_padding = array<i32: 0, 0>,
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0>,
    // CHECK-SAME: stride = array<i32: 2, 2>
    // CHECK-SAME: -> tensor<1x128x64x64xf32>

    %0 = stablehlo.reverse %arg1, dims = [2, 3] : tensor<256x128x2x2xf32>
    %1 = stablehlo.convolution(%arg0, %0)
      dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1],
      window = {
        stride = [1, 1],
        pad = [[1, 1], [1, 1]],
        lhs_dilate = [2, 2],
        rhs_dilate = [1, 1]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x256x32x32xf32>, tensor<256x128x2x2xf32>) -> tensor<1x128x64x64xf32>
    return %1 : tensor<1x128x64x64xf32>
  }

  func.func @test_conv_transpose2d_stablehlo(%arg0: tensor<1x14x14x768xf32>, %arg1: tensor<16x16x3x768xf32>) -> tensor<1x224x224x3xf32> {
    // CHECK: "ttir.conv_transpose2d"
    // CHECK-SAME: dilation = array<i32: 1, 1>,
    // CHECK-SAME: groups = 1 : i32,
    // CHECK-SAME: output_padding = array<i32: 0, 0>,
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0>,
    // CHECK-SAME: stride = array<i32: 16, 16>
    // CHECK-SAME: (tensor<1x14x14x768xf32>, tensor<768x3x16x16xf32>)
    // CHECK-SAME: -> tensor<1x224x224x3xf32>
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f],
      window = {
        stride = [1, 1],
        pad = [[15, 15], [15, 15]],
        lhs_dilate = [16, 16],
        rhs_dilate = [1, 1]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x14x14x768xf32>, tensor<16x16x3x768xf32>) -> tensor<1x224x224x3xf32>
    return %0 : tensor<1x224x224x3xf32>
  }

  func.func @test_conv_transpose2d_stablehlo_with_reverse(%arg0: tensor<1x64x1x6400xf32>, %arg1: tensor<1x8x32x64xf32>) -> tensor<1x32x1x25600xf32> {
    // CHECK-NOT: ttir.reverse
    // CHECK: %[[CONV_T:[0-9]+]] = "ttir.conv_transpose2d"
    // CHECK-SAME: dilation = array<i32: 1, 1>,
    // CHECK-SAME: groups = 1 : i32,
    // CHECK-SAME: output_padding = array<i32: 0, 0>,
    // CHECK-SAME: padding = array<i32: 0, 2, 0, 2>,
    // CHECK-SAME: stride = array<i32: 1, 4>
    // CHECK-SAME: (tensor<1x64x1x6400xf32>, tensor<64x32x1x8xf32>)
    // CHECK-SAME: -> tensor<1x32x1x25600xf32>
    %0 = stablehlo.reverse %arg1, dims = [0, 1] : tensor<1x8x32x64xf32>
    %1 = stablehlo.convolution(%arg0, %0)
      dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1],
      window = {
        stride = [1, 1],
        pad = [[0, 0], [5, 5]],
        lhs_dilate = [1, 4],
        rhs_dilate = [1, 1]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x64x1x6400xf32>, tensor<1x8x32x64xf32>) -> tensor<1x32x1x25600xf32>
    return %1 : tensor<1x32x1x25600xf32>
  }
}
