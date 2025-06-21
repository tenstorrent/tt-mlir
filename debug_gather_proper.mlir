func.func @test_gather_basic(%input: tensor<100x128xf32>, %indices: tensor<4x1xi32>) -> tensor<4x128xf32> {
  %0 = ttir.empty() : tensor<4x128xf32>
  %result = "ttir.gather"(%input, %indices, %0) {
      offset_dims = array<i64: 1>,
      collapsed_slice_dims = array<i64: 0>,
      operand_batching_dims = array<i64>,
      start_indices_batching_dims = array<i64>,
      start_index_map = array<i64: 0>,
      index_vector_dim = 1 : si64,
      slice_sizes = array<i64: 1, 128>,
      indices_are_sorted = false
  } : (tensor<100x128xf32>, tensor<4x1xi32>, tensor<4x128xf32>) -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
