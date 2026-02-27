// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for broadcast operation

// Verify that given attribute `broadcast_dimensions` is valid.
module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK: error: 'ttir.broadcast' op Input tensor rank should match output tensor rank.
    %1 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64 : 1, 16>}> : (tensor<1x1x32xf32>) -> tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1) : (tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %3 : tensor<1x16x32xf32>
  }
}
