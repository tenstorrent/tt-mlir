// RUN: ttmlir-opt %s | FileCheck %s

module attributes {} {
  func.func @conv_transpose2d_simple(%arg0: tensor<4x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<4x10x10x256xbf16> {
    %0 = tensor.empty() : tensor<4x10x10x256xbf16>
    // CHECK: %[[C:.*]] = "ttir.conv_transpose2d"[[C:.*]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<4x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<4x10x10x256xbf16>) -> tensor<4x10x10x256xbf16>
    return %1 : tensor<4x10x10x256xbf16>
  }

  func.func @conv_transpose2d_stride(%arg0: tensor<1x16x32x256xbf16>, %arg1: tensor<256x256x8x8xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x38x132x256xbf16> {
    %0 = tensor.empty() : tensor<1x38x132x256xbf16>
    // CHECK: %[[C:.*]] = "ttir.conv_transpose2d"[[C:.*]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 2, 4>,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x16x32x256xbf16>, tensor<256x256x8x8xbf16>, tensor<1x1x1x256xbf16>, tensor<1x38x132x256xbf16>) -> tensor<1x38x132x256xbf16>
    return %1 : tensor<1x38x132x256xbf16>
  }

  func.func @conv_transpose2d_padding(%arg0: tensor<1x64x64x256xbf16>, %arg1: tensor<256x256x16x16xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x73x67x256xbf16> {
    %0 = tensor.empty() : tensor<1x73x67x256xbf16>
    // CHECK: %[[C:.*]] = "ttir.conv_transpose2d"[[C:.*]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 3, 6>,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x64x64x256xbf16>, tensor<256x256x16x16xbf16>, tensor<1x1x1x256xbf16>, tensor<1x73x67x256xbf16>) -> tensor<1x73x67x256xbf16>
    return %1 : tensor<1x73x67x256xbf16>
  }

  func.func @conv_transpose2d_output_padding(%arg0: tensor<1x32x32x128xbf16>, %arg1: tensor<128x256x8x8xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x45x47x256xbf16> {
    %0 = tensor.empty() : tensor<1x45x47x256xbf16>
    // CHECK: %[[C:.*]] = "ttir.conv_transpose2d"[[C:.*]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = array<i32: 6, 8>,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x32x32x128xbf16>, tensor<128x256x8x8xbf16>, tensor<1x1x1x256xbf16>, tensor<1x45x47x256xbf16>) -> tensor<1x45x47x256xbf16>
    return %1 : tensor<1x45x47x256xbf16>
  }

  func.func @conv_transpose2d_dilation(%arg0: tensor<1x32x32x128xbf16>, %arg1: tensor<128x256x16x32xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x77x94x256xbf16> {
    %0 = tensor.empty() : tensor<1x77x94x256xbf16>
    // CHECK: %[[C:.*]] = "ttir.conv_transpose2d"[[C:.*]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = array<i32: 3, 2>,
              groups = 1: i32}
            > : (tensor<1x32x32x128xbf16>, tensor<128x256x16x32xbf16>, tensor<1x1x1x256xbf16>, tensor<1x77x94x256xbf16>) -> tensor<1x77x94x256xbf16>
    return %1 : tensor<1x77x94x256xbf16>
  }

  func.func @conv_transpose2d_groups(%arg0: tensor<1x16x32x192xbf16>, %arg1: tensor<192x126x8x8xbf16>, %arg2: tensor<1x1x1x252xbf16>) -> tensor<1x23x39x252xbf16> {
    %0 = tensor.empty() : tensor<1x23x39x252xbf16>
    // CHECK: %[[C:.*]] = "ttir.conv_transpose2d"[[C:.*]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 2: i32}
            > : (tensor<1x16x32x192xbf16>, tensor<192x126x8x8xbf16>, tensor<1x1x1x252xbf16>, tensor<1x23x39x252xbf16>) -> tensor<1x23x39x252xbf16>
    return %1 : tensor<1x23x39x252xbf16>
  }

  func.func @conv_transpose2d(%arg0: tensor<1x8x8x256xbf16>, %arg1: tensor<256x64x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x21x38x256xbf16> {
    %0 = tensor.empty() : tensor<1x21x38x256xbf16>
    // CHECK: %[[C:.*]] = "ttir.conv_transpose2d"[[C:.*]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = array<i32: 2, 3>,
              padding = array<i32: 6, 4>,
              output_padding = array<i32: 10, 12>,
              dilation = array<i32: 4, 6>,
              groups = 4: i32}
            > : (tensor<1x8x8x256xbf16>, tensor<256x64x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<1x21x38x256xbf16>) -> tensor<1x21x38x256xbf16>
    return %1 : tensor<1x21x38x256xbf16>
  }
}
