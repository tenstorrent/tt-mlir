module @jit_gather attributes {} {
  func.func public @test_gather(%operand: tensor<32000x1024xf32>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xf32> {
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1024>}> : (tensor<32000x1024xf32>, tensor<1x32xi32>) -> tensor<1x32x1024xf32>
    return %0 : tensor<1x32x1024xf32>
  }
}
