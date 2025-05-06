// RUN: ttmlir-opt --ttir-to-ttmetal-backend-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// UNSUPPORTED: true

module attributes {} {
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
