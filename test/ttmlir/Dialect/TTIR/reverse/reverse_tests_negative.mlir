// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for reverse operation

// Verify that parsing fails if dimensions are not unique.
module attributes {} {
  func.func @reverse_non_unique_dims(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    // CHECK: error: 'ttir.reverse' op dimensions should be unique. Got: 0, 0
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 0, 0>}> : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }
}

// Verify that parsing fails if any dimension is negative.
// -----
module attributes {} {
  func.func @reverse_negative_dim(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    // CHECK: error: 'ttir.reverse' op all dimensions should be non-negative. Got dimension: -1
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 0, -1>}> : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }
}

// Verify that parsing fails if any dimension is out of range [0, operandRank).
// -----
module attributes {} {
  func.func @reverse_out_of_bounds_dim(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    // CHECK: error: 'ttir.reverse' op all dimensions should be in interval [0, 2). Got dimension: 2
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 2>}> : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }
}
