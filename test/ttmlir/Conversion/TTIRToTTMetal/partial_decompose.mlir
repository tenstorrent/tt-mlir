// RUN: ttmlir-opt --ttir-to-ttir-decomposition="config=ttnn" -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=TTNN
// RUN: ttmlir-opt --ttir-to-ttir-decomposition="config=cpu-fallback" -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=CPU
// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=DEFAULT

module {
  func.func @test_decomposition_configs(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xf32>, %arg2: tensor<4x32xi32>) -> (tensor<32x1xbf16>, tensor<64x128xf32>, tensor<4x32x32xf32>, tensor<32x1xbf16>) {
    // Test reduce-or - should be decomposed in all configs
    %0 = ttir.empty() : tensor<32x1xbf16>
    // TTNN-NOT: ttir.reduce_or
    // CPU-NOT: ttir.reduce_or
    // DEFAULT-NOT: ttir.reduce_or
    %1 = "ttir.reduce_or"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x32xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>

    // Test dot-general - should be decomposed in all configs
    %arg3 = ttir.empty() : tensor<32x128xf32>
    %arg4 = ttir.empty() : tensor<64x32xf32>
    // TTNN-NOT: ttir.dot_general
    // CPU-NOT: ttir.dot_general
    // DEFAULT-NOT: ttir.dot_general
    %2 = "ttir.dot_general"(%arg4, %arg3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<64x32xf32>, tensor<32x128xf32>) -> tensor<64x128xf32>

    // Test gather - should be decomposed for TTNN, but NOT for CPU fallback
    %3 = ttir.empty() : tensor<4x32x32xf32>
    // TTNN-NOT: ttir.gather
    // CPU: ttir.gather
    // DEFAULT-NOT: ttir.gather
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

    // Test reduce-and - should be decomposed in all configs
    %5 = ttir.empty() : tensor<32x1xbf16>
    // TTNN-NOT: ttir.reduce_and
    // CPU-NOT: ttir.reduce_and
    // DEFAULT-NOT: ttir.reduce_and
    %6 = "ttir.reduce_and"(%arg0, %5) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x32xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>

    return %1, %2, %4, %6 : tensor<32x1xbf16>, tensor<64x128xf32>, tensor<4x32x32xf32>, tensor<32x1xbf16>
  }
}
