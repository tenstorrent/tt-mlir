// RUN: ttmlir-opt --ttir-to-ttir-decomposition="ops=reduce-or,gather" %s | FileCheck %s --check-prefix=PARTIAL
// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s --check-prefix=ALL
// RUN: ttmlir-opt --ttir-to-ttir-decomposition="ops=dot-general" %s | FileCheck %s --check-prefix=DOT-ONLY

module {
  func.func @test_selective_decomposition(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xf32>, %arg2: tensor<4x32xi32>) -> (tensor<32x1xbf16>, tensor<64x128xf32>, tensor<4x32x32xf32>) {
    // Test reduce-or - should be decomposed with ops=reduce-or
    %0 = ttir.empty() : tensor<32x1xbf16>
    // PARTIAL-NOT: ttir.reduce_or
    // PARTIAL: ttir.sum
    // ALL-NOT: ttir.reduce_or
    // DOT-ONLY: ttir.reduce_or
    %1 = "ttir.reduce_or"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x32xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>

    // Test dot-general - should NOT be decomposed with ops=reduce-or,gather
    %arg3 = ttir.empty() : tensor<32x128xf32>
    %arg4 = ttir.empty() : tensor<64x32xf32>
    // PARTIAL: ttir.dot_general
    // ALL-NOT: ttir.dot_general
    // DOT-ONLY-NOT: ttir.dot_general
    %2 = "ttir.dot_general"(%arg4, %arg3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<64x32xf32>, tensor<32x128xf32>) -> tensor<64x128xf32>

    // Test gather - should be decomposed with ops=reduce-or,gather
    %3 = ttir.empty() : tensor<4x32x32xf32>
    // PARTIAL-NOT: ttir.gather
    // PARTIAL: ttir.permute
    // PARTIAL: ttir.reshape
    // ALL-NOT: ttir.gather
    // DOT-ONLY: ttir.gather
    %4 = "ttir.gather"(%arg1, %arg2, %3) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64>,
        start_indices_batching_dims = array<i64>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 32>,
        indices_are_sorted = false
    } : (tensor<32x32xf32>, tensor<4x32xi32>, tensor<4x32x32xf32>) -> tensor<4x32x32xf32>

    return %1, %2, %4 : tensor<32x1xbf16>, tensor<64x128xf32>, tensor<4x32x32xf32>
  }
}
