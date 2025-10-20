// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xbf16>, %arg1: tensor<1x32x128x128xbf16>) -> tensor<1x32x64x64xbf16> {
    %0 = ttir.empty() : tensor<1x32x64x64xbf16>
    %1 = ttir.empty() : tensor<1x32x64x64xbf16>
    // CHECK-LABEL: func.func @forward
    // CHECK-NOT: "ttir.pooling"
    // CHECK: "ttnn.permute"(%arg0)
    // CHECK: "ttnn.view"
    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.max_pool2d"
    // CHECK: "ttnn.permute"(%arg1)
    // CHECK: "ttnn.view"
    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.max_pool2d"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.view"
    // CHECK: "ttnn.permute"
    // CHECK: return
    %2, %3 = "ttir.pooling"(%arg0, %arg1, %0, %1) <{
        operandSegmentSizes = array<i32: 2, 2>,
        pooling_method = #ttir<pooling_method Max>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>, tensor<1x32x64x64xbf16>, tensor<1x32x64x64xbf16>) -> (tensor<1x32x64x64xbf16>, tensor<1x32x64x64xbf16>)

    %4 = ttir.empty() : tensor<1x32x64x64xbf16>
    %5 = "ttir.add"(%2, %3, %4) : (tensor<1x32x64x64xbf16>, tensor<1x32x64x64xbf16>, tensor<1x32x64x64xbf16>) -> tensor<1x32x64x64xbf16>
    return %5 : tensor<1x32x64x64xbf16>
  }
  func.func @forward_mixed_types(%arg0: tensor<1x32x128x128xbf16>, %arg1: tensor<1x32x128x128xf32>) -> (tensor<1x32x64x64xbf16>, tensor<1x32x64x64xf32>) {
    // CHECK-LABEL: func.func @forward_mixed_types
    // CHECK: "ttnn.permute"(%arg0)
    // CHECK: "ttnn.max_pool2d"
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.permute"(%arg1)
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.max_pool2d"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.permute"
    // CHECK: return
    %0 = ttir.empty() : tensor<1x32x64x64xbf16>
    %1 = ttir.empty() : tensor<1x32x64x64xf32>
    %2, %3 = "ttir.pooling"(%arg0, %arg1, %0, %1) <{
        operandSegmentSizes = array<i32: 2, 2>,
        pooling_method = #ttir<pooling_method Max>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>
      }> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xf32>, tensor<1x32x64x64xbf16>, tensor<1x32x64x64xf32>)
        -> (tensor<1x32x64x64xbf16>, tensor<1x32x64x64xf32>)
    return %2, %3 : tensor<1x32x64x64xbf16>, tensor<1x32x64x64xf32>
  }
}
