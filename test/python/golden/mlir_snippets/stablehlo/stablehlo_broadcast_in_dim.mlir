module @jit_broadcast_in_dim attributes {} {
  func.func public @test_broadcast_1d_to_2d(%arg0: tensor<3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func public @test_broadcast_scalar(%arg0: tensor<f32>) -> tensor<4x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
  func.func public @test_broadcast_with_expansion(%arg0: tensor<2x4xf32>) -> tensor<2x3x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<2x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
  func.func public @test_broadcast_2d_to_4d(%arg0: tensor<8x16xf32>) -> tensor<4x2x8x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 3] : (tensor<8x16xf32>) -> tensor<4x2x8x16xf32>
    return %0 : tensor<4x2x8x16xf32>
  }
}
