// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for NOperands trait

// CHECK: error: 'ttir.abs' op expected 2 operands, but found 3
func.func @eltwise_unary(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensor.empty() : tensor<64x64xbf16>
  %1 = "ttir.abs"(%arg0, %arg0, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %1 : tensor<64x64xbf16>
}

// -----
// CHECK: error: 'ttir.add' op expected 3 operands, but found 4
func.func @eltwise_binary(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = tensor.empty() : tensor<64x64xf32>
  %1 = "ttir.add"(%arg0, %arg1, %arg1, %0) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// -----
// CHECK: error: 'ttir.add' op expected 3 operands, but found 2
func.func @eltwise_binary(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = tensor.empty() : tensor<64x64xf32>
  %1 = "ttir.add"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// -----
// CHECK: error: 'ttir.where' op expected 4 operands, but found 5
func.func @eltwise_ternary(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = tensor.empty() : tensor<64x64xf32>
  %1 = "ttir.where"(%arg0, %arg1, %arg2, %arg2, %0) <{operandSegmentSizes = array<i32: 4, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}
