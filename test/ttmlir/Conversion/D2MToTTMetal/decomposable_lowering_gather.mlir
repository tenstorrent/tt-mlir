// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// UNSUPPORTED: true
// Full lowering requires:
//   Embedding: https://github.com/tenstorrent/tt-mlir/issues/3024

module {
  func.func @test_gather(%operand: tensor<32x32xf32>, %start_indices: tensor<4x32xi32>) -> tensor<4x32x32xf32> {
    %0 = ttir.empty() : tensor<4x32x32xf32>
    // CHECK-NOT: ttir.gather
    // CHECK: embedding
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 32>,
        indices_are_sorted = false
    } : (tensor<32x32xf32>, tensor<4x32xi32>, tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    return %1 : tensor<4x32x32xf32>
  }
}
