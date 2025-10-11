// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

module  {
  func.func @linear_with_batched_rhs_and_bias(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<2x1024x1024xf32>, %arg2: tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>{
    // CHECK-LABEL: func.func @linear_with_batched_rhs_and_bias
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<2x33x1024xf32
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<2x33x1024xf32
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x33x1024xf32>, tensor<2x1024x1024xf32>, tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>
    return %result : tensor<2x33x1024xf32>
  }

  func.func @linear_bias_broadcast(%arg0: tensor<4x3x64x128xbf16>, %arg1: tensor<4x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    // CHECK-LABEL: func.func @linear_bias_broadcast
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<4x3x64x32xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<14x4x3x64x32xbf16
    %result = "ttnn.linear"(%arg0, %arg1, %bias) : (tensor<4x3x64x128xbf16>, tensor<4x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %result : tensor<14x4x3x64x32xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_matmul(%arg0: tensor<1x3x64x128xbf16>, %arg1: tensor<1x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    // CHECK-LABEL: func.func @linear_nd_nd_bias_broadcast_matmul
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<1x3x64x32xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<14x4x3x64x32xbf16
    %result = "ttnn.linear"(%arg0, %arg1, %bias) : (tensor<1x3x64x128xbf16>, tensor<1x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %result : tensor<14x4x3x64x32xbf16>
  }
}
