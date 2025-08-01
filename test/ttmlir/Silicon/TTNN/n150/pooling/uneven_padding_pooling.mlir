// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module attributes {} {
  func.func @test_maxpool2d_uneven_padding(%arg0: tensor<1x32x126x126xf32>) -> tensor<1x32x64x64xf32> {
    // CHECK-LABEL: @test_maxpool2d
    %0 = ttir.empty() : tensor<1x32x64x64xf32>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK: "ttnn.pad"
    // CHECK: "ttnn.max_pool2d"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    %1 = "ttir.pooling"(%arg0, %0) <{
        operandSegmentSizes = array<i32: 1, 1>,
        pooling_method = #ttir<pooling_method Max>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 2, 0, 2>}> : (tensor<1x32x126x126xf32>, tensor<1x32x64x64xf32>) -> tensor<1x32x64x64xf32>
    return %1 : tensor<1x32x64x64xf32>
  }
}

module attributes {} {
  func.func @test_maxpool2d_uneven_padding_ceil_mode(%arg0: tensor<1x32x124x128xf32>) -> tensor<1x32x64x64xf32> {
    // CHECK-LABEL: @test_maxpool2d
    %0 = ttir.empty() : tensor<1x32x64x64xf32>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK: "ttnn.pad"
    // CHECK: "ttnn.max_pool2d"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    %1 = "ttir.pooling"(%arg0, %0) <{
        operandSegmentSizes = array<i32: 1, 1>,
        pooling_method = #ttir<pooling_method Max>,
        window_dimensions = array<i64: 1, 1, 2, 2>,
        window_strides = array<i64: 1, 1, 2, 2>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 1, 3, 0, 0>}> : (tensor<1x32x124x128xf32>, tensor<1x32x64x64xf32>) -> tensor<1x32x64x64xf32>
    return %1 : tensor<1x32x64x64xf32>
  }
}
