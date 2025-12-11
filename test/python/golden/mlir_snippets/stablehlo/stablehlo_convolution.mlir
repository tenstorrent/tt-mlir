module @jit_convolution attributes {} {
  func.func public @test_convolution(%arg0: tensor<1x128x128x32xf32>, %arg1: tensor<64x32x3x3xf32>) -> tensor<1x128x128x64xf32> {
    %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f],
      window = {
        stride = [1, 1],
        pad = [[1, 1], [1, 1]],
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64
      } : (tensor<1x128x128x32xf32>, tensor<64x32x3x3xf32>) -> tensor<1x128x128x64xf32>
    return %0 : tensor<1x128x128x64xf32>
  }
}
