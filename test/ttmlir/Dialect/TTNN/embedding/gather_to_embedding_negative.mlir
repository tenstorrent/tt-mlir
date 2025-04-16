// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s 2>&1  | FileCheck %s
// Negative tests for gather to embedding conversion

// Verify that the parsing fails if there are operand and start indices batching dimensions
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<2x8x7xf32>, %start_indices: tensor<2x3xi32>) -> tensor<2x3x7xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<2x3x7xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 1>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 1>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 1, 7>, // ??
        indices_are_sorted = false
    } : (tensor<2x8x7xf32>, tensor<2x3xi32>, tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
    return %1 : tensor<2x3x7xf32>
  }
}
