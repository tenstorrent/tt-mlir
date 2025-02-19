// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for concat operation

// Verify that verification fails when the given dimension is out of bounds and is negative.
module {
  func.func @concat_negative_dim_out_of_bounds_negative(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
    // CHECK: error: 'ttir.concat' op Invalid dimension -3 for concatenation.
    %0 = tensor.empty() : tensor<32x96xf32>
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = -3 : si32}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
    return %1 : tensor<32x96xf32>
  }
}
// -----

// Verify that verification fails when the given dimension is out of bounds and is positive.
module {
  func.func @concat_negative_dim_out_of_bounds_positive(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
    // CHECK: error: 'ttir.concat' op Invalid dimension 2 for concatenation.
    %0 = tensor.empty() : tensor<32x96xf32>
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 2 : si32}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
    return %1 : tensor<32x96xf32>
  }
}

// -----

// Verify that verification fails if all input tensors doesn't have the same rank.
module {
  func.func @forward(%arg0: tensor<32x32xf32>, %arg1: tensor<1x32x64xf32>) -> tensor<32x96xf32> {
    // CHECK: error: 'ttir.concat' op All input tensors must have the same rank.
    %0 = tensor.empty() : tensor<32x96xf32>
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 0 : si32}> : (tensor<32x32xf32>, tensor<1x32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
    return %1 : tensor<32x96xf32>
  }
}

// -----

// Verify that verification fails if all input tensors have same rank but only the specified dim can have different size.
module {
  func.func @forward(%arg0: tensor<32x1x64xf32>, %arg1: tensor<32x64x2xf32>) -> tensor<32x65x64xf32> {
    // CHECK: error: 'ttir.concat' op All input tensors must have the same dimensions, except for dimension 1.
    %0 = tensor.empty() : tensor<32x96xf32>
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32}> : (tensor<32x1x64xf32>, tensor<32x64x2xf32>, tensor<32x96xf32>) -> tensor<32x65x64xf32>
    return %1 : tensor<32x65x64xf32>
  }
}
