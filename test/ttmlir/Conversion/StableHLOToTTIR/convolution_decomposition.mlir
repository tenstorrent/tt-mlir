// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @test_convolution {
  // Test 1D convolution sliced with batch_group_count > 1. Each slice is a 1D
  // conv, so it is routed to ttir.conv1d.
  func.func @test_conv_sliced(%arg0: tensor<1x9x3072xbf16>, %arg1: tensor<1x9x768xbf16>) -> tensor<1x768x768xbf16> {
    // CHECK-COUNT-8: "ttir.slice_static"
    // CHECK-COUNT-4: "ttir.conv1d"
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

  // Test depthwise convolution (feature_group_count > 1, not transposed).
  // Genuine 2D (both spatial dims > 1) must stay conv2d.
  func.func @test_conv_not_transposed(%arg0: tensor<1x2048x16x16xbf16>, %arg1: tensor<2048x1x3x3xbf16>) -> tensor<1x2048x16x16xbf16> {
    // CHECK-LABEL: @test_conv_not_transposed
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

  // Test a conv1d that the framework (e.g. torch-xla) lowered to a 2D
  // convolution with a size-1 spatial dim. It must be routed to ttir.conv1d
  // (so it lowers to ttnn.conv1d and can use an L1 config, avoiding the
  // in-DRAM depthwise conv2d hang).
  func.func @test_degenerate_2d_to_conv1d(%arg0: tensor<1x8x1x20xbf16>, %arg1: tensor<8x1x1x4xbf16>) -> tensor<1x8x1x23xbf16> {
    // CHECK-LABEL: @test_degenerate_2d_to_conv1d
    // CHECK: "ttir.conv1d"
    // CHECK-NOT: "ttir.conv2d"
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {
        stride = [1, 1],
        pad = [[0, 0], [3, 3]]
      } {
        feature_group_count = 8 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x8x1x20xbf16>, tensor<8x1x1x4xbf16>) -> tensor<1x8x1x23xbf16>
    return %0 : tensor<1x8x1x23xbf16>
  }

  // Test a genuine 2D pointwise (1x1) conv on a [N,C,1,1] tensor, as produced
  // by the squeeze-and-excitation blocks in EfficientNet/VovNet (a global
  // average pool feeding a 1x1 conv). BOTH spatial dims are size 1, so there
  // is no real 1D axis; it must stay a conv2d and NOT be routed to conv1d
  // (which would move the weight parameter to host and break trace hoisting,
  // tt-mlir #9076).
  func.func @test_pointwise_2d_not_conv1d(%arg0: tensor<1x8x1x1xbf16>, %arg1: tensor<16x8x1x1xbf16>) -> tensor<1x16x1x1xbf16> {
    // CHECK-LABEL: @test_pointwise_2d_not_conv1d
    // CHECK: "ttir.conv2d"
    // CHECK-NOT: "ttir.conv1d"
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {
        stride = [1, 1],
        pad = [[0, 0], [0, 0]]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x8x1x1xbf16>, tensor<16x8x1x1xbf16>) -> tensor<1x16x1x1xbf16>
    return %0 : tensor<1x16x1x1xbf16>
  }
}
