// REQUIRES: stablehlo
// RUN: not ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline %s 2>&1  | FileCheck %s
// Negative tests for gather to embedding conversion

// Verify that the parsing fails if there are operand and start indices batching dimensions
// -----
module attributes {} {
  func.func @negative_batching(%operand: tensor<2x8x7xf32>, %start_indices: tensor<2x3xi32>) -> tensor<2x3x7xf32> {
    // CHECK: error: failed to legalize operation 'stablehlo.gather'
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 7>}> : (tensor<2x8x7xf32>, tensor<2x3xi32>) -> tensor<2x3x7xf32>
    return %0 : tensor<2x3x7xf32>
  }
}

// Verify that the parsing fails if we are indexing multiple dims, but startIndices is 1D
// -----
module attributes {} {
  func.func @negative_1D_start_indices(%operand: tensor<8x7x3xf32>, %start_indices: tensor<2xi32>) -> tensor<3xf32> {
    // CHECK: error: failed to legalize operation 'stablehlo.gather'
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 0>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 3>}> : (tensor<8x7x3xf32>, tensor<2xi32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
