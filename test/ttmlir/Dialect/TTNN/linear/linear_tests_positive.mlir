// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @linear_1d_1d_bias(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>, %bias: tensor<1xbf16>) -> tensor<1xbf16> {
    %0 = ttir.empty() : tensor<1xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<1xbf16
    // CHECK-SAME: tensor<1xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

    func.func @linear_1d_1d_bias_broadcast(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>, %bias: tensor<128xbf16>) -> tensor<128xbf16> {
    %0 = ttir.empty() : tensor<128xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<128xbf16>
    return %1 : tensor<128xbf16>
  }

  func.func @linear_2d_2d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = ttir.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

    // linear nd - nd tests
    // Batched linear is decomposed into matmul + add.
  func.func @linear_nd_nd_bias_broadcast_bias(%arg0: tensor<14x7x32x32xbf16>, %arg1:tensor<14x1x32x64xbf16>, %bias: tensor<64xbf16>) -> tensor<14x7x32x64xbf16> {
    %0 = ttir.empty() : tensor<14x7x32x64xbf16>
    // CHECK: "ttnn.matmul"(%arg0, %arg1)
    // CHECK: "ttnn.add"(%{{.*}}, %arg2)
    // CHECK: -> tensor<14x4x3x64x32xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<14x7x32x32xbf16>, tensor<14x1x32x64xbf16>, tensor<64xbf16>, tensor<14x7x32x64xbf16>) -> tensor<14x7x32x64xbf16>
    return %1 : tensor<14x7x32x64xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_matmul(%arg0: tensor<3x64x128xbf16>, %arg1: tensor<4x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    %0 = ttir.empty() : tensor<14x4x3x64x32xbf16>
    // CHECK: "ttnn.matmul"(%arg0, %arg1)
    // CHECK: "ttnn.add"(%{{.*}}, %arg2)
    // CHECK: -> tensor<14x4x3x64x32xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<3x64x128xbf16>, tensor<4x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %1 : tensor<14x4x3x64x32xbf16>
  }

  // Linear with transposed inputs tests.
  func.func @linear_2d_tranpose_2d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %0 = ttir.empty() : tensor<128x128xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = false
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x128xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) <{transpose_a = true}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }
  func.func @linear_2d_2d_transpose_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = ttir.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: transpose_a = false
    // CHECK-SAME: transpose_b = true
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) <{transpose_b = true}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
  func.func @linear_2d_tranpose_2d_transpose(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %0 = ttir.empty() : tensor<128x128xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = true
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x64xbf16
    // CHECK-SAME: tensor<128x128xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) <{transpose_a = true, transpose_b = true}> : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }
}
