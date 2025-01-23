// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for broadcast operation

// Verify that each input shape matches output shape.
module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK: error: 'ttir.broadcast' op Input tensor rank of 2 does not match output tensor rank of 3
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i32 : 1, 16, 1>}> : (tensor<1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %3 : tensor<1x16x32xf32>
  }
}
