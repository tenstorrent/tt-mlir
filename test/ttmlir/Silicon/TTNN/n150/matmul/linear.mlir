// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @linear(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32> {
    %0 = ttir.empty() : tensor<2x34x1024xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %1 : tensor<2x34x1024xf32>
  }
  func.func @linear_with_implicit_broadcast(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<1024xf32>) -> tensor<2x34x1024xf32> {
    %0 = ttir.empty() : tensor<2x34x1024xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %1 : tensor<2x34x1024xf32>
  }

  func.func @linear_with_implicit_broadcast_2(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<2x2x34x1024xf32>) -> tensor<2x2x34x1024xf32> {
    %0 = ttir.empty() : tensor<2x2x34x1024xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<2x34x1024xf32>, tensor<1024x1024xf32>, tensor<2x2x34x1024xf32>, tensor<2x2x34x1024xf32>) -> tensor<2x2x34x1024xf32>
    return %1 : tensor<2x2x34x1024xf32>
  }

  func.func @linear_with_batched_rhs_and_bias(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<2x1024x1024xf32>, %bias: tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32> {
    %0 = ttir.empty() : tensor<2x33x1024xf32>
    // this will be lowered to a matmul + add
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<2x33x1024xf32>, tensor<2x1024x1024xf32>, tensor<2x33x1024xf32>, tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>
    return %1 : tensor<2x33x1024xf32>
  }

  func.func @linear_bias_broadcast(%arg0: tensor<4x3x64x128xbf16>, %arg1: tensor<4x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    %0 = ttir.empty() : tensor<14x4x3x64x32xbf16>
    // this will be lowered to a matmul + add
    // Bias broadcasts from [14, 4, 3, 64, 32] (adds leading batch dim 14)
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<4x3x64x128xbf16>, tensor<4x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %1 : tensor<14x4x3x64x32xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast(%arg0: tensor<1x1x64x128xbf16>, %arg1: tensor<1x1x128x32xbf16>, %bias: tensor<4x3x64x32xbf16>) -> tensor<4x3x64x32xbf16> {
    %0 = ttir.empty() : tensor<4x3x64x32xbf16>
    // The expected output shape is [1, 1, 64, 32] with leading batch dims broadcasted to [4, 3, 64, 32] due to bias.
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<1x1x64x128xbf16>, tensor<1x1x128x32xbf16>, tensor<4x3x64x32xbf16>, tensor<4x3x64x32xbf16>) -> tensor<4x3x64x32xbf16>
    return %1 : tensor<4x3x64x32xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_matmul(%arg0: tensor<1x3x64x128xbf16>, %arg1: tensor<1x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    %0 = ttir.empty() : tensor<14x4x3x64x32xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<1x3x64x128xbf16>, tensor<1x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %1 : tensor<14x4x3x64x32xbf16>
  }

  func.func @linear_1d_lhs(%arg0: tensor<1024xf32>, %arg1: tensor<1x1024x1024xf32>, %bias: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
    %0 = ttir.empty() : tensor<1x1024xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0): (tensor<1024xf32>, tensor<1x1024x1024xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    return %1 : tensor<1x1024xf32>
  }

  func.func @linear_with_1d_rhs(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<1024xf32>, %bias: tensor<2x33xf32>) -> tensor<2x33xf32> {
    %0 = ttir.empty() : tensor<2x33xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<2x33x1024xf32>, tensor<1024xf32>, tensor<2x33xf32>, tensor<2x33xf32>) -> tensor<2x33xf32>
    return %1 : tensor<2x33xf32>
  }
}
