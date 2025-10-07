// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

module  {
  func.func @test_linear_op_rewrite(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<2x1024x1024xf32>, %arg2: tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>{
    // CHECK-LABEL: func.func @test_linear_op_rewrite
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.add"
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x33x1024xf32>, tensor<2x1024x1024xf32>, tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>
    return %result : tensor<2x33x1024xf32>
  }
  // linear nd - nd tests
  func.func @linear_nd_nd_bias_broadcast_bias(%arg0: tensor<14x7x32x32xbf16>, %arg1:tensor<14x1x32x64xbf16>, %bias: tensor<64xbf16>) -> tensor<14x7x32x64xbf16> {
    %0 = ttir.empty() : tensor<14x7x32x64xbf16>
    %1 = "ttnn.linear"(%arg0, %arg1, %bias) : (tensor<14x7x32x32xbf16>, tensor<14x1x32x64xbf16>, tensor<64xbf16>) -> tensor<14x7x32x64xbf16>
    return %1 : tensor<14x7x32x64xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_matmul(%arg0: tensor<3x64x128xbf16>, %arg1: tensor<4x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    %0 = ttir.empty() : tensor<14x4x3x64x32xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<3x64x128xbf16>, tensor<4x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %1 : tensor<14x4x3x64x32xbf16>
  }
}
