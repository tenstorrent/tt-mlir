// RUN: ttmlir-opt --ttir-to-ttir-decomposition --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module @test_gather_repeat_front {
  func.func @repeat_front(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x6x60x106xbf16>) {
    // CHECK: ttir.slice_static
    // CHECK: ttir.repeat
    // CHECK: ttir.concat
    // CHECK-NOT: ttir.gather
    %1 = "ttir.constant"() <{value = dense<[[0], [0], [0], [1], [2], [3]]> : tensor<6x1xi64>}> : () -> tensor<6x1xi64>
    %2 = "ttir.gather"(%arg0, %1) <{collapsed_slice_dims = array<i64: 2>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 0, 1, 3, 4>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 768, 1, 60, 106>, start_index_map = array<i64: 2>, start_indices_batching_dims = array<i64>}> : (tensor<1x768x4x60x106xbf16>, tensor<6x1xi64>) -> tensor<1x768x6x60x106xbf16>
    return %2 : tensor<1x768x6x60x106xbf16>
  }
}

module @test_gather_repeat_back {
  func.func @repeat_back(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x6x60x106xbf16>) {
    // CHECK: ttir.slice_static
    // CHECK: ttir.repeat
    // CHECK: ttir.concat
    // CHECK-NOT: ttir.gather
    %1 = "ttir.constant"() <{value = dense<[[0], [1], [2], [3], [3], [3]]> : tensor<6x1xi64>}> : () -> tensor<6x1xi64>
    %2 = "ttir.gather"(%arg0, %1) <{collapsed_slice_dims = array<i64: 2>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 0, 1, 3, 4>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 768, 1, 60, 106>, start_index_map = array<i64: 2>, start_indices_batching_dims = array<i64>}> : (tensor<1x768x4x60x106xbf16>, tensor<6x1xi64>) -> tensor<1x768x6x60x106xbf16>
    return %2 : tensor<1x768x6x60x106xbf16>
  }
}

module @test_gather_slice_no_repeat_both {
  func.func @repeat_both(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x6x60x106xbf16>) {
    // CHECK: ttir.slice_static
    // CHECK: ttir.slice_static
    // CHECK: ttir.concat
    // CHECK-NOT: ttir.gather
    %1 = "ttir.constant"() <{value = dense<[[0], [0], [1], [2], [3], [3]]> : tensor<6x1xi64>}> : () -> tensor<6x1xi64>
    %2 = "ttir.gather"(%arg0, %1) <{collapsed_slice_dims = array<i64: 2>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 0, 1, 3, 4>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 768, 1, 60, 106>, start_index_map = array<i64: 2>, start_indices_batching_dims = array<i64>}> : (tensor<1x768x4x60x106xbf16>, tensor<6x1xi64>) -> tensor<1x768x6x60x106xbf16>
    return %2 : tensor<1x768x6x60x106xbf16>
  }
}


module @test_gather_repeat_both_multiple {
  func.func @repeat_both_multiple(%arg0: tensor<1x768x4x60x106xbf16>) -> (tensor<1x768x8x60x106xbf16>) {
    // CHECK: ttir.slice_static
    // CHECK: ttir.repeat
    // CHECK: ttir.slice_static
    // CHECK: ttir.repeat
    // CHECK: ttir.concat
    // CHECK-NOT: ttir.gather
    %1 = "ttir.constant"() <{value = dense<[[0], [0], [0], [1], [2], [3], [3], [3]]> : tensor<8x1xi64>}> : () -> tensor<8x1xi64>
    %2 = "ttir.gather"(%arg0, %1) <{collapsed_slice_dims = array<i64: 2>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 0, 1, 3, 4>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 768, 1, 60, 106>, start_index_map = array<i64: 2>, start_indices_batching_dims = array<i64>}> : (tensor<1x768x4x60x106xbf16>, tensor<8x1xi64>) -> tensor<1x768x8x60x106xbf16>
    return %2 : tensor<1x768x8x60x106xbf16>
  }
}
