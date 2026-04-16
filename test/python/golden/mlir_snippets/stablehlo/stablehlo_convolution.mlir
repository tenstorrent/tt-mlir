module @jit_convolution attributes {} {
  func.func public @test_convolution(%arg0: tensor<1x3x8x8xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<1x16x6x6xf32> {
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {
        stride = [1, 1],
        pad = [[0, 0], [0, 0]],
        lhs_dilate = [1, 1],
        rhs_dilate = [1, 1]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x6x6xf32>
    return %0 : tensor<1x16x6x6xf32>
  }
}
