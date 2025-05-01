// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for Broadcastable interface

// CHECK: error: 'ttir.abs' op result shape (2, 64) doesn't match expected shape after broadcasting (1, 64)
func.func @eltwise_unary(%arg0: tensor<1x64xbf16>) -> tensor<2x64xbf16> {
  %0 = ttir.empty() : tensor<2x64xbf16>
  %1 = "ttir.abs"(%arg0, %0) : (tensor<1x64xbf16>, tensor<2x64xbf16>) -> tensor<2x64xbf16>
  return %1 : tensor<2x64xbf16>
}

// -----
// CHECK: error: 'ttir.multiply' op operand shape (4, 64, 128) is not broadcast compatible with inferred operand shapes (2, 64, 128)
func.func @eltwise_binary_non_compatible_operands(%arg0: tensor<2x64x128xf32>, %arg1: tensor<4x64x128xf32>) -> tensor<4x64x128xf32> {
  %0 = ttir.empty() : tensor<4x64x128xf32>
  %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<2x64x128xf32>, tensor<4x64x128xf32>, tensor<4x64x128xf32>) -> tensor<4x64x128xf32>
  return %1 : tensor<4x64x128xf32>
}

// -----
// CHECK: error: 'ttir.add' op result shape (4, 2, 3, 64) doesn't match expected shape after broadcasting (2, 3, 64)
func.func @eltwise_binary(%arg0: tensor<2x3x64xf32>, %arg1: tensor<64xf32>) -> tensor<4x2x3x64xf32> {
  %0 = ttir.empty() : tensor<4x2x3x64xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x3x64xf32>, tensor<64xf32>, tensor<4x2x3x64xf32>) -> tensor<4x2x3x64xf32>
  return %1 : tensor<4x2x3x64xf32>
}

// -----
// CHECK: error: 'ttir.where' op result shape (1, 2, 3, 64) doesn't match expected shape after broadcasting (2, 3, 64)
func.func @eltwise_ternary(%arg0: tensor<3x64xf32>, %arg1: tensor<1x3x64xf32>, %arg2: tensor<2x1x64xf32>) -> tensor<1x2x3x64xf32> {
  %0 = ttir.empty() : tensor<1x2x3x64xf32>
  %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<3x64xf32>, tensor<1x3x64xf32>, tensor<2x1x64xf32>, tensor<1x2x3x64xf32>) -> tensor<1x2x3x64xf32>
  return %1 : tensor<1x2x3x64xf32>
}

// -----
// CHECK: error: 'ttir.where' op operand shape (3, 64, 128) is not broadcast compatible with inferred operand shapes (2, 64, 128)
func.func @eltwise_ternary_non_compatible_operands(%arg0: tensor<64x128xf32>, %arg1: tensor<2x64x128xf32>, %arg2: tensor<3x64x128xf32>) -> tensor<3x64x128xf32> {
  %0 = ttir.empty() : tensor<3x64x128xf32>
  %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<64x128xf32>, tensor<2x64x128xf32>, tensor<3x64x128xf32>, tensor<3x64x128xf32>) -> tensor<3x64x128xf32>
  return %1 : tensor<3x64x128xf32>
}
