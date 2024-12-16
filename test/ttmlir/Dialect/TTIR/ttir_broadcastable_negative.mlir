// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for Broadcastable interface

// CHECK: 'ttir.abs' op Result shape must match operand shapes after broadcasting

func.func @eltwise_unary(%arg0: tensor<1x64xbf16>) -> tensor<2x64xbf16> {
  %0 = tensor.empty() : tensor<2x64xbf16>
  %1 = "ttir.abs"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64xbf16>, tensor<2x64xbf16>) -> tensor<2x64xbf16>
  return %1 : tensor<2x64xbf16>
}

// -----
// CHECK: error: 'ttir.add' op Result shape must match operand shapes after broadcasting

func.func @eltwise_binary(%arg0: tensor<2x3x64xf32>, %arg1: tensor<64xf32>) -> tensor<4x2x3x64xf32> {
  %0 = tensor.empty() : tensor<4x2x3x64xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2x3x64xf32>, tensor<64xf32>, tensor<4x2x3x64xf32>) -> tensor<4x2x3x64xf32>
  return %1 : tensor<4x2x3x64xf32>
}

// -----
// CHECK: error: 'ttir.where' op Result shape must match operand shapes after broadcasting

func.func @eltwise_ternary(%arg0: tensor<3x64xf32>, %arg1: tensor<1x3x64xf32>, %arg2: tensor<2x1x64xf32>) -> tensor<1x2x3x64xf32> {
  %0 = tensor.empty() : tensor<1x2x3x64xf32>
  %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<3x64xf32>, tensor<1x3x64xf32>, tensor<2x1x64xf32>, tensor<1x2x3x64xf32>) -> tensor<1x2x3x64xf32>
  return %1 : tensor<1x2x3x64xf32>
}
