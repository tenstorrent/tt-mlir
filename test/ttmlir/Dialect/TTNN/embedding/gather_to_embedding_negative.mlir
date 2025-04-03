// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s 2>&1  | FileCheck %s
// Negative tests for gather to embedding conversion

// Verify that the parsing fails if start_index_map isn't [0]
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<4x3xf32>, %start_indices: tensor<1x3xi32>) -> tensor<1x3x4xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<1x3x4xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 1>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 1>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 4, 1>,
        indices_are_sorted = false
    } : (tensor<4x3xf32>, tensor<1x3xi32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    return %1 : tensor<1x3x4xf32>
  }
}

// Verify that the parsing fails if operand has more than 2 non-unit dimensions
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<4x3x4xf32>, %start_indices: tensor<1x3xi32>) -> tensor<1x3x3x4xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<1x3x3x4xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2,3>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1,3,4>,
        indices_are_sorted = false
    } : (tensor<4x3x4xf32>, tensor<1x3xi32>, tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    return %1 : tensor<1x3x3x4xf32>
  }
}

// Verify that the parsing fails if sliceSizes can't be reshaped to [1, fullDim]
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<4x3xf32>, %start_indices: tensor<1x3xi32>) -> tensor<1x3x1xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<1x3x1xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1,1>,
        indices_are_sorted = false
    } : (tensor<4x3xf32>, tensor<1x3xi32>, tensor<1x3x1xf32>) -> tensor<1x3x1xf32>
    return %1 : tensor<1x3x1xf32>
  }
}

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

// Verify that the parsing fails if there is a non-unit dimension after index_vector_dim in start indices
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<3x4xf32>, %start_indices: tensor<1x3xi32>) -> tensor<3x4xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<3x4xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 1>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 0 : si64,
        slice_sizes = array<i64: 1, 4>,
        indices_are_sorted = false
    } : (tensor<3x4xf32>, tensor<1x3xi32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %1 : tensor<3x4xf32>
  }
}

// Verify that the parsing fails if there are more than two non-unit dimensions in start indices
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<15x4xf32>, %start_indices: tensor<2x2x3xi32>) -> tensor<2x2x3x4xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<2x2x3x4xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 3>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 3 : si64,
        slice_sizes = array<i64: 1, 4>,
        indices_are_sorted = false
    } : (tensor<15x4xf32>, tensor<2x2x3xi32>, tensor<2x2x3x4xf32>) -> tensor<2x2x3x4xf32>
    return %1 : tensor<2x2x3x4xf32>
  }
}

// Verify that the parsing fails if offset dimension isn't in the same place as index_vector_dim
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<3x4xf32>, %start_indices: tensor<1x3xi32>) -> tensor<4x1x3xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = ttir.empty() : tensor<4x1x3xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 0>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 4>,
        indices_are_sorted = false
    } : (tensor<3x4xf32>, tensor<1x3xi32>, tensor<4x1x3xf32>) -> tensor<4x1x3xf32>
    return %1 : tensor<4x1x3xf32>
  }
}