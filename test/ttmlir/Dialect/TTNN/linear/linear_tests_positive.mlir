// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @linear_1d_1d_bias(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>, %bias: tensor<1xbf16>) -> tensor<1xbf16> {
    // CHECK-LABEL: func.func @linear_1d_1d_bias
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<128xbf16
    // CHECK-SAME: tensor<1xbf16
    // CHECK-SAME: tensor<1xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<128xbf16>, tensor<128xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  // Bias last dim (128) != matmul output last dim (1) for vector-vector product,
  // so decomposed into matmul + add.
  func.func @linear_1d_1d_bias_broadcast(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>, %bias: tensor<128xbf16>) -> tensor<128xbf16> {
    // CHECK-LABEL: func.func @linear_1d_1d_bias_broadcast
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.add"
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<128xbf16>
    return %1 : tensor<128xbf16>
  }

  // Bias is 2D with non-unit non-feature dim (64x64), so decomposed into matmul + add.
  func.func @linear_2d_2d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK-LABEL: func.func @linear_2d_2d_bias
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.add"
    %1 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  // linear nd - nd tests
  // Batched linear is decomposed into matmul + add.
  func.func @linear_nd_nd_bias_broadcast_bias(%arg0: tensor<14x7x32x32xbf16>, %arg1:tensor<14x1x32x64xbf16>, %bias: tensor<64xbf16>) -> tensor<14x7x32x64xbf16> {
    // CHECK-LABEL: func.func @linear_nd_nd_bias_broadcast_bias
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.add"
    %0 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<14x7x32x32xbf16>, tensor<14x1x32x64xbf16>, tensor<64xbf16>) -> tensor<14x7x32x64xbf16>
    return %0 : tensor<14x7x32x64xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_matmul(%arg0: tensor<3x64x128xbf16>, %arg1: tensor<4x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    // CHECK-LABEL: func.func @linear_nd_nd_bias_broadcast_matmul
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<14x4x3x64x32xbf16
    %0 = "ttir.linear"(%arg0, %arg1, %bias) : (tensor<3x64x128xbf16>, tensor<4x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %0 : tensor<14x4x3x64x32xbf16>
  }

  // Linear with transposed inputs tests.
  func.func @linear_2d_tranpose_1d_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<128xbf16>) -> tensor<128x128xbf16> {
    // CHECK-LABEL: func.func @linear_2d_tranpose_1d_bias
    // CHECK: "ttnn.linear"
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = false
    %0 = "ttir.linear"(%arg0, %arg1, %bias) <{transpose_a = true}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<128xbf16>) -> tensor<128x128xbf16>
    return %0 : tensor<128x128xbf16>
  }

  func.func @linear_2d_1d_transpose_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %bias: tensor<64xbf16>) -> tensor<64x64xbf16> {
    // CHECK-LABEL: func.func @linear_2d_1d_transpose_bias
    // CHECK: "ttnn.linear"
    // CHECK-SAME: transpose_a = false
    // CHECK-SAME: transpose_b = true
    %0 = "ttir.linear"(%arg0, %arg1, %bias) <{transpose_b = true}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }

  func.func @linear_2d_tranpose_2d_transpose(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<128xbf16>) -> tensor<128x128xbf16> {
    // CHECK-LABEL: func.func @linear_2d_tranpose_2d_transpose
    // CHECK: "ttnn.linear"
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = true
    %0 = "ttir.linear"(%arg0, %arg1, %bias) <{transpose_a = true, transpose_b = true}> : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<128xbf16>) -> tensor<128x128xbf16>
    return %0 : tensor<128x128xbf16>
  }

  func.func @main_batch_linear_with_bias_right_transpose(%arg_a : tensor<12x24x64xf32>, %arg_b : tensor<12x24x64xf32>, %bias : tensor<12x24x24xf32>) -> tensor<12x24x24xf32>{
    // CHECK-LABEL: func.func @main_batch_linear_with_bias_right_transpose
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: transpose_a = false
    // CHECK-SAME: transpose_b = true
    // CHECK: "ttnn.add"
    %0 = "ttir.linear"(%arg_a, %arg_b, %bias) <{transpose_a = false, transpose_b = true}> : (tensor<12x24x64xf32>, tensor<12x24x64xf32>, tensor<12x24x24xf32>) -> tensor<12x24x24xf32>
    return %0 : tensor<12x24x24xf32>
  }

  func.func @main_batch_linear_with_bias_left_transpose(%arg_a : tensor<12x24x64xf32>, %arg_b : tensor<12x24x64xf32>, %bias : tensor<12x64x64xf32>) -> tensor<12x64x64xf32>{
    // CHECK-LABEL: func.func @main_batch_linear_with_bias_left_transpose
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = false
    // CHECK: "ttnn.add"
    %0 = "ttir.linear"(%arg_a, %arg_b, %bias) <{transpose_a = true, transpose_b = false}> : (tensor<12x24x64xf32>, tensor<12x24x64xf32>, tensor<12x64x64xf32>) -> tensor<12x64x64xf32>
    return %0 : tensor<12x64x64xf32>
  }
}
