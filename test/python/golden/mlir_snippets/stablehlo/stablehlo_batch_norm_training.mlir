module @jit_batch_norm_training attributes {} {
  func.func public @test_batch_norm_training(%operand: tensor<2x4x8x8xf32>, %scale: tensor<4xf32>, %offset: tensor<4xf32>) -> (tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>) {
    %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%operand, %scale, %offset) {
      epsilon = 1.0e-5 : f32,
      feature_index = 1 : i64
    } : (tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>)
    return %output, %batch_mean, %batch_var : tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
