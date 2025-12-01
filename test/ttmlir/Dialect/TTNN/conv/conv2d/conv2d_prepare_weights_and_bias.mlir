// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @conv2d_no_bias_bf16(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1) <{dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16>
    return %0 : tensor<1x30x30x64xbf16>
  }

  func.func @conv2d_with_bias_bf16(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.prepare_conv2d_bias"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2) <{dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %0 : tensor<1x30x30x64xbf16>
  }

  func.func @conv2d_no_bias_f32(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>) -> tensor<1x30x30x64xf32> {
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1) <{dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<1x32x32x64xf32>, tensor<64x64x3x3xf32>) -> tensor<1x30x30x64xf32>
    return %0 : tensor<1x30x30x64xf32>
  }

  func.func @conv2d_with_bias_f32(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<1x1x1x64xf32>) -> tensor<1x30x30x64xf32> {
    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK: "ttnn.prepare_conv2d_bias"
    // CHECK: "ttnn.conv2d"
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2) <{dilation = 1 : i32, groups = 1 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<1x32x32x64xf32>, tensor<64x64x3x3xf32>, tensor<1x1x1x64xf32>) -> tensor<1x30x30x64xf32>
    return %0 : tensor<1x30x30x64xf32>
  }
}
