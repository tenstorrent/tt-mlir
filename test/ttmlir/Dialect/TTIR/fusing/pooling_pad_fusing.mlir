// RUN: ttmlir-opt -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @main(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x56x56xbf16>{
    // CHECK-NOT: "ttir.pad"
    // CHECK: padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>,
    %0 = ttir.empty() : tensor<1x64x114x114xbf16>
    %1 = "ttir.pad"(%arg0, %0) <{padding = array<i32: 0, 0, 0, 0, 1, 1, 1, 1>, value = 0xFF800000 : f32}> : (tensor<1x64x112x112xbf16>, tensor<1x64x114x114xbf16>) -> tensor<1x64x114x114xbf16>
    %2 = ttir.empty() : tensor<1x64x56x56xbf16>
    %3 = "ttir.pooling"(%1, %2) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<1x64x114x114xbf16>, tensor<1x64x56x56xbf16>) -> tensor<1x64x56x56xbf16>
    return %3: tensor<1x64x56x56xbf16>
  }
}
