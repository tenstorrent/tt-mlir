module @jit_broadcast_in_dim attributes {} {
  func.func public @test_broadcast_in_dim(%arg0: tensor<3x1x5xf32>) -> tensor<3x4x1x5x6xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<3x1x5xf32>) -> tensor<3x4x1x5x6xf32>
    return %0 : tensor<3x4x1x5x6xf32>
  }
}
