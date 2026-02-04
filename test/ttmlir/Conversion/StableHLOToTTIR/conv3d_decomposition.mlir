// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @test_convolution3d {

  // Test 3D convolution with different layout (NCDHW -> NDHWC conversion)
  func.func @test_conv3d_layout_conversion(%arg0: tensor<1x4x8x28x28xbf16>, %arg1: tensor<16x4x3x3x3xbf16>) -> tensor<1x16x6x26x26xbf16> {
    // CHECK: "ttir.conv3d"
    // CHECK-SAME: batch_dim = 0 : i64
    // CHECK-SAME: channel_dim = 1 : i64
    // CHECK-SAME: depth_dim = 2 : i64
    // CHECK-SAME: height_dim = 3 : i64
    // CHECK-SAME: width_dim = 4 : i64

    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2],
      window = {
        stride = [1, 1, 1],
        pad = [[0, 0], [0, 0], [0, 0]]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x4x8x28x28xbf16>, tensor<16x4x3x3x3xbf16>) -> tensor<1x16x6x26x26xbf16>
    return %0 : tensor<1x16x6x26x26xbf16>
  }

  // Test 3D convolution with stride and padding
  func.func @test_conv3d_stride_padding(%arg0: tensor<1x16x8x28x28xbf16>, %arg1: tensor<32x16x3x3x3xbf16>) -> tensor<1x32x4x14x14xbf16> {
    // CHECK: "ttir.conv3d"
    // CHECK-SAME: batch_dim = 0 : i64
    // CHECK-SAME: channel_dim = 1 : i64
    // CHECK-SAME: depth_dim = 2 : i64
    // CHECK-SAME: height_dim = 3 : i64
    // CHECK-SAME: width_dim = 4 : i64

    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2],
      window = {
        stride = [2, 2, 2],
        pad = [[1, 1], [1, 1], [1, 1]]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x16x8x28x28xbf16>, tensor<32x16x3x3x3xbf16>) -> tensor<1x32x4x14x14xbf16>
    return %0 : tensor<1x32x4x14x14xbf16>
  }
}
