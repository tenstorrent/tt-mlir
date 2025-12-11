module {
  func.func @gather_wrapper(%arg0: tensor<100x50xf32>) -> tensor<10x50xf32> {
    %0 = "ttir.zeros"() <{shape = array<i32: 10>}> : () -> tensor<10xf32>
    %1 = "ttir.gather"(%arg0, %0) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 1>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 50>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<100x50xf32>, tensor<10xf32>) -> tensor<10x50xf32>
    return %1 : tensor<10x50xf32>
  }
}
