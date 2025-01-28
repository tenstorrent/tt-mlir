// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s 2>&1  | FileCheck %s
// Negative tests for gather to embedding conversion

// Verify that the parsing fails if the slice_sizes.size <= 1
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<32000x1024xf32>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = tensor.empty() : tensor<1x32x1024xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 1 : si64,
        slice_sizes = array<i64: 1>,
        indices_are_sorted = false
    } : (tensor<32000x1024xf32>, tensor<1x32xi32>, tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    return %1 : tensor<1x32x1024xf32>
  }
}

// Verify that the parsing fails if the slice_sizes.size != [1, hiddenDim]
// -----
module attributes {} {
  func.func @negative_slice_sizes_1(%operand: tensor<32000x1024xf32>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xf32> {
    %0 = tensor.empty() : tensor<1x32x1024xf32>
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 1 : si64,
        slice_sizes = array<i64: 1, 384>,
        indices_are_sorted = false
    } : (tensor<32000x1024xf32>, tensor<1x32xi32>, tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    return %1 : tensor<1x32x1024xf32>
  }
}

// Verify that the parsing fails if the offsetDims != [2]
// -----
module attributes {} {
  func.func @negative_slice_sizes_0(%operand: tensor<32000x1024xf32>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xf32> {
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %0 = tensor.empty() : tensor<1x32x1024xf32>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 3>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 1 : si64,
        slice_sizes = array<i64: 1, 384>,
        indices_are_sorted = false
    } : (tensor<32000x1024xf32>, tensor<1x32xi32>, tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    return %1 : tensor<1x32x1024xf32>
  }
}

// Verify that the parsing fails if collapsed_slice_dims != [0]
// -----
module attributes {} {
  func.func @negative_collapsed_slice_dims(%operand: tensor<32000x1024xf32>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xf32> {
    %0 = tensor.empty() : tensor<1x32x1024xf32>
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 1>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 1 : si64,
        slice_sizes = array<i64: 1, 1024>,
        indices_are_sorted = false
    } : (tensor<32000x1024xf32>, tensor<1x32xi32>, tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    return %1 : tensor<1x32x1024xf32>
  }
}

// Verify that the parsing fails slice_indices != 1 when slice_indices.size == output.size
// -----
module attributes {} {
  func.func @negative_start_indices(%operand: tensor<448x384xf32>, %start_indices: tensor<1x2x2xi32>) -> tensor<1x2x384xf32> {
    %0 = tensor.empty() : tensor<1x2x384xf32>
    // CHECK: error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 384>,
        indices_are_sorted = false
      }> : (tensor<448x384xf32>, tensor<1x2x2xi32>, tensor<1x2x384xf32>) -> tensor<1x2x384xf32>
    return %1 : tensor<1x2x384xf32>
  }
}
