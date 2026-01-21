// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @test_convolution {
  // Test 1D convolution sliced with batch_group_count > 1
  func.func @test_conv_sliced(%arg0: tensor<1x9x3072xbf16>, %arg1: tensor<1x9x768xbf16>) -> tensor<1x768x768xbf16> {
    // CHECK-COUNT-8: "ttir.slice_static"
    // CHECK-COUNT-4: "ttir.conv2d"
    // CHECK: "ttir.concat"
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [f, 0, b]x[i, 0, o]->[0, b, f],
      window = {
        stride = [1],
        pad = [[0, 0]]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 4 : i64
      } : (tensor<1x9x3072xbf16>, tensor<1x9x768xbf16>) -> tensor<1x768x768xbf16>
    return %0 : tensor<1x768x768xbf16>
  }

  // Test 2D convolution sliced with batch_group_count > 1
  func.func @test_conv2d_sliced(%arg0: tensor<2x4x32x32xbf16>, %arg1: tensor<16x4x3x3xbf16>) -> tensor<1x16x32x32xbf16> {
    // CHECK-COUNT-4: "ttir.slice_static"
    // CHECK-COUNT-2: "ttir.conv2d"
    // CHECK: "ttir.concat"
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {
        stride = [1, 1],
        pad = [[1, 1], [1, 1]]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 2 : i64
      } : (tensor<2x4x32x32xbf16>, tensor<16x4x3x3xbf16>) -> tensor<1x16x32x32xbf16>
    return %0 : tensor<1x16x32x32xbf16>
  }

  // Test depthwise convolution (feature_group_count > 1, not transposed)
  func.func @test_conv_not_transposed(%arg0: tensor<1x2048x16x16xbf16>, %arg1: tensor<2048x1x3x3xbf16>) -> tensor<1x2048x16x16xbf16> {
    // CHECK: "ttir.conv2d"
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {
        stride = [1, 1],
        pad = [[1, 1], [1, 1]]
      } {
        feature_group_count = 2048 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x2048x16x16xbf16>, tensor<2048x1x3x3xbf16>) -> tensor<1x2048x16x16xbf16>
    return %0 : tensor<1x2048x16x16xbf16>
  }
}
