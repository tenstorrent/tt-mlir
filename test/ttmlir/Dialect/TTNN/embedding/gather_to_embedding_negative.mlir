// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s 2>&1  | FileCheck %s
// Negative tests for gather to embedding conversion

// Verify that the parsing fails if there are operand and start indices batching dimensions
// -----
module attributes {} {
  func.func @negative_batching(%operand: tensor<2x8x7xf32>, %start_indices: tensor<2x3xi32>) -> tensor<2x3x7xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<2x3x7xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 1>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 1>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 1, 7>,
        indices_are_sorted = false
    } : (tensor<2x8x7xf32>, tensor<2x3xi32>, tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
    return %1 : tensor<2x3x7xf32>
  }
}

// Verify that the parsing fails if we are indexing multiple dims, but startIndices is 1D
// -----
module attributes {} {
  func.func @negative_1D_start_indices(%operand: tensor<8x7x3xf32>, %start_indices: tensor<2xi32>) -> tensor<3xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<3xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 0>,
        collapsed_slice_dims = array<i64: 0, 1>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0, 1>,
        index_vector_dim = 0 : si64,
        slice_sizes = array<i64: 1, 1, 3>,
        indices_are_sorted = false
    } : (tensor<8x7x3xf32>, tensor<2xi32>, tensor<3xf32>) -> tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
