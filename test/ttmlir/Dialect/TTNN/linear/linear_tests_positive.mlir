// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @linear_1d_1d(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>) -> tensor<1xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<1xbf16
    %0 = tensor.empty() : tensor<1xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<1xbf16
    // CHECK-SAME: tensor<1xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  func.func @linear_1d_1d_bias(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>, %bias: tensor<1xbf16>) -> tensor<1xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<1xbf16
    %0 = tensor.empty() : tensor<1xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<1xbf16
    // CHECK-SAME: tensor<1xbf16
    // CHECK-SAME: tensor<1xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

    func.func @linear_1d_1d_bias_broadcast(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>, %bias: tensor<128xbf16>) -> tensor<128xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<128xbf16
    %0 = tensor.empty() : tensor<128xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<128xbf16>
    return %1 : tensor<128xbf16>
  }

  func.func @linear_2d_1d(%arg0: tensor<64x128xbf16>, %arg1: tensor<128xbf16>) -> tensor<64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<64xbf16
    %0 = tensor.empty() : tensor<64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<64xbf16
    // CHECK-SAME: tensor<64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    return %1 : tensor<64xbf16>
  }

  func.func @linear_2d_2d(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<64x64xbf16
    %0 = tensor.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

    func.func @linear_2d_2d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<64x64xbf16
    %0 = tensor.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @linear_1d_nd(%arg0: tensor<128xbf16>, %arg1: tensor<12x7x128x64xbf16>) -> tensor<12x7x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<12x7x64xbf16
    %0 = tensor.empty() : tensor<12x7x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<12x7x128x64xbf16
    // CHECK-SAME: tensor<12x7x64xbf16
    // CHECK-SAME: tensor<12x7x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<12x7x128x64xbf16>, tensor<12x7x64xbf16>) -> tensor<12x7x64xbf16>
    return %1 : tensor<12x7x64xbf16>
  }

  func.func @linear_nd_1d(%arg0: tensor<12x7x128x64xbf16>, %arg1: tensor<64xbf16>) -> tensor<12x7x128xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<12x7x128xbf16
    %0 = tensor.empty() : tensor<12x7x128xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<12x7x128x64xbf16
    // CHECK-SAME: tensor<64xbf16
    // CHECK-SAME: tensor<12x7x128xbf16
    // CHECK-SAME: tensor<12x7x128xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<12x7x128x64xbf16>, tensor<64xbf16>, tensor<12x7x128xbf16>) -> tensor<12x7x128xbf16>
    return %1 : tensor<12x7x128xbf16>
  }

  func.func @linear_2d_nd(%arg0: tensor<64x128xbf16>, %arg1: tensor<12x7x128x64xbf16>) -> tensor<12x7x64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<12x7x64x64xbf16
    %0 = tensor.empty() : tensor<12x7x64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<12x7x128x64xbf16
    // CHECK-SAME: tensor<12x7x64x64xbf16
    // CHECK-SAME: tensor<12x7x64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<12x7x128x64xbf16>, tensor<12x7x64x64xbf16>) -> tensor<12x7x64x64xbf16>
    return %1 : tensor<12x7x64x64xbf16>
  }

  func.func @linear_nd_2d(%arg0: tensor<12x7x128x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<12x7x128x128xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<12x7x128x128xbf16
    %0 = tensor.empty() : tensor<12x7x128x128xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<12x7x128x64xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<12x7x128x128xbf16
    // CHECK-SAME: tensor<12x7x128x128xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<12x7x128x64xbf16>, tensor<64x128xbf16>, tensor<12x7x128x128xbf16>) -> tensor<12x7x128x128xbf16>
    return %1 : tensor<12x7x128x128xbf16>
  }

  // linear nd - nd tests
  func.func @linear_nd_nd_same_rank_same_dims(%arg0: tensor<7x64x128xbf16>, %arg1: tensor<7x128x64xbf16>) -> tensor<7x64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<7x64x64xbf16
    %0 = tensor.empty() : tensor<7x64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<7x64x128xbf16
    // CHECK-SAME: tensor<7x128x64xbf16
    // CHECK-SAME: tensor<7x64x64xbf16
    // CHECK-SAME: tensor<7x64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<7x64x128xbf16>, tensor<7x128x64xbf16>, tensor<7x64x64xbf16>) -> tensor<7x64x64xbf16>
    return %1 : tensor<7x64x64xbf16>
  }

  func.func @linear_nd_nd_same_rank_broadcastable_dims_1(%arg0: tensor<7x64x128xbf16>, %arg1: tensor<1x128x64xbf16>) -> tensor<7x64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<7x64x64xbf16
    %0 = tensor.empty() : tensor<7x64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<7x64x128xbf16
    // CHECK-SAME: tensor<1x128x64xbf16
    // CHECK-SAME: tensor<7x64x64xbf16
    // CHECK-SAME: tensor<7x64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<7x64x128xbf16>, tensor<1x128x64xbf16>, tensor<7x64x64xbf16>) -> tensor<7x64x64xbf16>
    return %1 : tensor<7x64x64xbf16>
  }

  func.func @linear_nd_nd_same_rank_broadcastable_dims_2(%arg0: tensor<1x7x64x128xbf16>, %arg1: tensor<7x1x128x64xbf16>) -> tensor<7x7x64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<7x7x64x64xbf16
    %0 = tensor.empty() : tensor<7x7x64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<1x7x64x128xbf16
    // CHECK-SAME: tensor<7x1x128x64xbf16
    // CHECK-SAME: tensor<7x7x64x64xbf16
    // CHECK-SAME: tensor<7x7x64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<1x7x64x128xbf16>, tensor<7x1x128x64xbf16>, tensor<7x7x64x64xbf16>) -> tensor<7x7x64x64xbf16>
    return %1 : tensor<7x7x64x64xbf16>
  }

  func.func @linear_nd_nd_different_rank_broadcastable_dims_2(%arg0: tensor<12x1x7x64x128xbf16>, %arg1: tensor<7x1x128x64xbf16>) -> tensor<12x7x7x64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<12x7x7x64x64xbf16
    %0 = tensor.empty() : tensor<12x7x7x64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<12x1x7x64x128xbf16
    // CHECK-SAME: tensor<7x1x128x64xbf16
    // CHECK-SAME: tensor<12x7x7x64x64xbf16
    // CHECK-SAME: tensor<12x7x7x64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<12x1x7x64x128xbf16>, tensor<7x1x128x64xbf16>, tensor<12x7x7x64x64xbf16>) -> tensor<12x7x7x64x64xbf16>
    return %1 : tensor<12x7x7x64x64xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_bias(%arg0: tensor<14x7x32x32xbf16>, %arg1:tensor<14x1x32x64xbf16>, %bias: tensor<64xbf16>) -> tensor<14x7x32x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<14x7x32x64xbf16
    %0 = tensor.empty() : tensor<14x7x32x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<14x7x32x32xbf16
    // CHECK-SAME: tensor<14x1x32x64xbf16
    // CHECK-SAME: tensor<64xbf16
    // CHECK-SAME: tensor<14x7x32x64xbf16
    // CHECK-SAME: tensor<14x7x32x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<14x7x32x32xbf16>, tensor<14x1x32x64xbf16>, tensor<64xbf16>, tensor<14x7x32x64xbf16>) -> tensor<14x7x32x64xbf16>
    return %1 : tensor<14x7x32x64xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_matmul(%arg0: tensor<3x64x128xbf16>, %arg1: tensor<4x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<14x4x3x64x32xbf16
    %0 = tensor.empty() : tensor<14x4x3x64x32xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<3x64x128xbf16
    // CHECK-SAME: tensor<4x3x128x32xbf16
    // CHECK-SAME: tensor<14x4x3x64x32xbf16
    // CHECK-SAME: tensor<14x4x3x64x32xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<3x64x128xbf16>, tensor<4x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %1 : tensor<14x4x3x64x32xbf16>
  }
}
