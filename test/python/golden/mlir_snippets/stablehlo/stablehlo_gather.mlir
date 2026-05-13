module @jit_gather attributes {} {
  func.func public @test_gather_simple(%operand: tensor<8x4xf32>) -> tensor<3x4xf32> {
    %indices = stablehlo.constant dense<[0, 2, 5]> : tensor<3xi32>
    %0 = "stablehlo.gather"(%operand, %indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<8x4xf32>, tensor<3xi32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }

  func.func public @test_gather_embedding(%operand: tensor<16x8xf32>) -> tensor<1x4x8xf32> {
    %indices = stablehlo.constant dense<[[0, 3, 7, 12]]> : tensor<1x4xi32>
    %0 = "stablehlo.gather"(%operand, %indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<16x8xf32>, tensor<1x4xi32>) -> tensor<1x4x8xf32>
    return %0 : tensor<1x4x8xf32>
  }

  func.func public @test_gather_complex(%operand: tensor<8x4x8xf32>) -> tensor<2x2x4xf32> {
    %indices = stablehlo.constant dense<[[[0, 1], [2, 3]], [[1, 0], [3, 2]]]> : tensor<2x2x2xi32>
    %0 = "stablehlo.gather"(%operand, %indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0, 2], start_index_map = [0, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4, 1>}> : (tensor<8x4x8xf32>, tensor<2x2x2xi32>) -> tensor<2x2x4xf32>
    return %0 : tensor<2x2x4xf32>
  }
}
