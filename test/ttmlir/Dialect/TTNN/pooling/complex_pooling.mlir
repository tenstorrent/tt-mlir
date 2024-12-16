// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xbf16>, %arg1: tensor<1x32x128x128xbf16>) -> tensor<1x32x64x64xbf16> {
    %0 = tensor.empty() : tensor<1x32x64x64xbf16>
    %1 = tensor.empty() : tensor<1x32x64x64xbf16>
    // CHECK: %[[C:.*]] = "ttnn.max_pool2d"[[C:.*]]
    %2, %3 = "ttir.pooling"(%arg0, %arg1, %0, %1) <{
        operandSegmentSizes = array<i32: 2, 2>,
        pooling_method = #ttir<pooling_method Max>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>, tensor<1x32x64x64xbf16>, tensor<1x32x64x64xbf16>) -> (tensor<1x32x64x64xbf16>, tensor<1x32x64x64xbf16>)

    %4 = tensor.empty() : tensor<1x32x64x64xbf16>
    %6 = "ttir.add"(%2, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x64x64xbf16>, tensor<1x32x64x64xbf16>, tensor<1x32x64x64xbf16>) -> tensor<1x32x64x64xbf16>
    return %6 : tensor<1x32x64x64xbf16>
  }
}
