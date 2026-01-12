module @jit_batch_norm_inference attributes {} {
  func.func public @test_batch_norm_inference(
    %operand: tensor<2x4x8x8xf32>,
    %scale: tensor<4xf32>,
    %offset: tensor<4xf32>,
    %mean: tensor<4xf32>,
    %variance: tensor<4xf32>
  ) -> tensor<2x4x8x8xf32> {

    %output = "stablehlo.batch_norm_inference"(
      %operand, %scale, %offset, %mean, %variance
    ) {
      epsilon = 1.0e-5 : f32,
      feature_index = 1 : i64
    } : (tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
      -> tensor<2x4x8x8xf32>

    return %output : tensor<2x4x8x8xf32>
  }
}
