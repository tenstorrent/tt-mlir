// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @conv_transpose2d_no_bias_bf16(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x34x34x64xbf16> {
    %0 = ttir.empty() : tensor<1x34x34x64xbf16>
    // CHECK: "ttnn.prepare_conv_transpose2d_weights"
    // CHECK: "ttnn.conv_transpose2d"
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, output_padding = array<i32: 0, 0>, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x34x34x64xbf16>) -> tensor<1x34x34x64xbf16>
    return %1 : tensor<1x34x34x64xbf16>
  }

  func.func @conv_transpose2d_with_bias_bf16(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x34x34x64xbf16> {
    %0 = ttir.empty() : tensor<1x34x34x64xbf16>
    // CHECK: "ttnn.prepare_conv_transpose2d_weights"
    // CHECK: "ttnn.prepare_conv_transpose2d_bias"
    // CHECK: "ttnn.conv_transpose2d"
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, output_padding = array<i32: 0, 0>, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x34x34x64xbf16>) -> tensor<1x34x34x64xbf16>
    return %1 : tensor<1x34x34x64xbf16>
  }

  func.func @conv_transpose2d_no_bias_f32(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>) -> tensor<1x34x34x64xf32> {
    %0 = ttir.empty() : tensor<1x34x34x64xf32>
    // CHECK: "ttnn.prepare_conv_transpose2d_weights"
    // CHECK: "ttnn.conv_transpose2d"
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, output_padding = array<i32: 0, 0>, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x32x32x64xf32>, tensor<64x64x3x3xf32>, tensor<1x34x34x64xf32>) -> tensor<1x34x34x64xf32>
    return %1 : tensor<1x34x34x64xf32>
  }

  func.func @conv_transpose2d_with_bias_f32(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<1x1x1x64xf32>) -> tensor<1x34x34x64xf32> {
    %0 = ttir.empty() : tensor<1x34x34x64xf32>
    // CHECK: "ttnn.prepare_conv_transpose2d_weights"
    // CHECK: "ttnn.prepare_conv_transpose2d_bias"
    // CHECK: "ttnn.conv_transpose2d"
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, output_padding = array<i32: 0, 0>, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x32x32x64xf32>, tensor<64x64x3x3xf32>, tensor<1x1x1x64xf32>, tensor<1x34x34x64xf32>) -> tensor<1x34x34x64xf32>
    return %1 : tensor<1x34x34x64xf32>
  }
}
