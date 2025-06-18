// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module attributes {} {
  func.func @test_maxpool2d() -> tensor<1x32x64x64xbf16> {
    // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 1, 32, 64, 64>}> : () -> tensor<1x32x64x64xbf16>
    // CHECK return %[[FULL]]
    %cst = "ttir.full"() <{fill_value = 1.0 : f32, shape = array<i32: 1, 32, 128, 128>}> : () -> tensor<1x32x128x128xbf16>
    %0 = ttir.empty() : tensor<1x32x64x64xbf16>
    %1 = "ttir.pooling"(%cst, %0) <{
        operandSegmentSizes = array<i32: 1, 1>,
        pooling_method = #ttir<pooling_method Max>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>}> : (tensor<1x32x128x128xbf16>, tensor<1x32x64x64xbf16>) -> tensor<1x32x64x64xbf16>
    return %1 : tensor<1x32x64x64xbf16>
  }

  func.func @test_avgpool2d() -> tensor<1x32x64x64xbf16> {
    // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 2.000000e+00 : f32, shape = array<i32: 1, 32, 64, 64>}> : () -> tensor<1x32x64x64xbf16>
    // CHECK return %[[FULL]]
    %cst = "ttir.full"() <{fill_value = 2.0 : f32, shape = array<i32: 1, 32, 128, 128>}> : () -> tensor<1x32x128x128xbf16>
    %0 = ttir.empty() : tensor<1x32x64x64xbf16>
    %1 = "ttir.pooling"(%cst, %0) <{
        operandSegmentSizes = array<i32: 1, 1>,
        pooling_method = #ttir<pooling_method Average>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>}> : (tensor<1x32x128x128xbf16>, tensor<1x32x64x64xbf16>) -> tensor<1x32x64x64xbf16>
    return %1 : tensor<1x32x64x64xbf16>
  }

  func.func @test_sumpool2d() -> tensor<1x32x64x64xbf16> {
    // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 4.000000e+00 : f32, shape = array<i32: 1, 32, 64, 64>}> : () -> tensor<1x32x64x64xbf16>
    // CHECK return %[[FULL]]
    %cst = "ttir.full"() <{fill_value = 1.0 : f32, shape = array<i32: 1, 32, 128, 128>}> : () -> tensor<1x32x128x128xbf16>
    %0 = ttir.empty() : tensor<1x32x64x64xbf16>
    %1 = "ttir.pooling"(%cst, %0) <{
        operandSegmentSizes = array<i32: 1, 1>,
        pooling_method = #ttir<pooling_method Sum>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>}> : (tensor<1x32x128x128xbf16>, tensor<1x32x64x64xbf16>) -> tensor<1x32x64x64xbf16>
    return %1 : tensor<1x32x64x64xbf16>
  }

  func.func @test_pool_multiple_inputs() -> (tensor<1x32x64x64xbf16>, tensor<1x32x10x10x10xbf16>) {
    // CHECK: %[[FULL0:.*]] = "ttir.full"() <{fill_value = 4.000000e+00 : f32, shape = array<i32: 1, 32, 64, 64>}> : () -> tensor<1x32x64x64xbf16>
    // CHECK: %[[FULL1:.*]] = "ttir.full"() <{fill_value = 8.000000e+00 : f32, shape = array<i32: 1, 32, 10, 10, 10>}> : () -> tensor<1x32x10x10x10xbf16>
    // CHECK return %[[FULL0]], %[[FULL1]]
    %cst0 = "ttir.full"() <{fill_value = 1.0 : f32, shape = array<i32: 1, 32, 128, 128>}> : () -> tensor<1x32x128x128xbf16>
    %cst1 = "ttir.full"() <{fill_value = 2.0 : f32, shape = array<i32: 1, 32, 20, 20>}> : () -> tensor<1x32x20x20xbf16>
    %0 = ttir.empty() : tensor<1x32x64x64xbf16>
    %1 = ttir.empty() : tensor<1x32x10x10x10xbf16>
    %2:2 = "ttir.pooling"(%cst0, %cst1, %0, %1) <{
        operandSegmentSizes = array<i32: 2, 2>,
        pooling_method = #ttir<pooling_method Sum>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>}> : (tensor<1x32x128x128xbf16>, tensor<1x32x20x20xbf16>, tensor<1x32x64x64xbf16>, tensor<1x32x10x10x10xbf16>) -> (tensor<1x32x64x64xbf16>, tensor<1x32x10x10x10xbf16>)
    return %2#0, %2#1 : tensor<1x32x64x64xbf16>, tensor<1x32x10x10x10xbf16>
  }
}
