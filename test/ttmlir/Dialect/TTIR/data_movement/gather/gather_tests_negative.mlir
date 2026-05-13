// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for gather operation

// Verify that index tensor must be unsigned integer type.
module {
  func.func @gather_signed_index(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xbf16>) -> tensor<2x3xf32> {
    // CHECK: error: 'ttir.gather' op Index tensor must have an integer type, got 'bf16'
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 0 : i32}> : (tensor<5x3xf32>, tensor<2x3xbf16>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Verify that input and index tensors must have the same rank.
module {
  func.func @gather_rank_mismatch(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3x1xui32>) -> tensor<2x3xf32> {
    // CHECK: error: 'ttir.gather' op Input tensor and index tensor must have the same rank.
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 0 : i32}> : (tensor<5x3xf32>, tensor<2x3x1xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Verify that dim is not out of range (too large).
module {
  func.func @gather_dim_too_large(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    // CHECK: error: 'ttir.gather' op Dimension must be in the range [-2, 2), got dim = 2
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = 2 : i32}> : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

// Verify that dim is not out of range (too negative).
module {
  func.func @gather_dim_too_negative(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    // CHECK: error: 'ttir.gather' op Dimension must be in the range [-2, 2), got dim = -3
    %0 = "ttir.gather"(%arg0, %arg1) <{dim = -3 : i32}> : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
