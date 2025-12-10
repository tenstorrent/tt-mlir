module @jit_batch_norm_grad attributes {} {
  func.func public @test_batch_norm_grad(%operand: tensor<2x4x8x8xf32>,
                                         %scale: tensor<4xf32>,
                                         %mean: tensor<4xf32>,
                                         %variance: tensor<4xf32>,
                                         %grad_output: tensor<2x4x8x8xf32>)
                                         -> (tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>) {
    %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
      epsilon = 1.0e-5 : f32,
      feature_index = 1 : i64
    } : (tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<2x4x8x8xf32>)
        -> (tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>)
    return %grad_operand, %grad_scale, %grad_offset : tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
