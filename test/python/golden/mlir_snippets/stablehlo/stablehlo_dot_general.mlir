module @jit_dot_general attributes {} {
  func.func public @test_matmul(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >
    } : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }

  func.func public @test_batched_matmul(%arg0: tensor<8x32x64xf32>, %arg1: tensor<8x64x128xf32>) -> tensor<8x32x128xf32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_batching_dimensions = [0],
        rhs_contracting_dimensions = [1]
      >
    } : (tensor<8x32x64xf32>, tensor<8x64x128xf32>) -> tensor<8x32x128xf32>
    return %0 : tensor<8x32x128xf32>
  }

  func.func public @test_multi_batch(%arg0: tensor<4x8x16x32xf32>, %arg1: tensor<4x8x32x64xf32>) -> tensor<4x8x16x64xf32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_batching_dimensions = [0, 1],
        rhs_contracting_dimensions = [2]
      >
    } : (tensor<4x8x16x32xf32>, tensor<4x8x32x64xf32>) -> tensor<4x8x16x64xf32>
    return %0 : tensor<4x8x16x64xf32>
  }

  func.func public @test_vector_dot(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<f32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [0],
        rhs_contracting_dimensions = [0]
      >
    } : (tensor<128xf32>, tensor<128xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
