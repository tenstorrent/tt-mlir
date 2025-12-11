module @jit_broadcast_in_dim attributes {} {
  func.func public @test_broadcast_in_dim(%operand: tensor<1x128xf32>) -> tensor<32x128xf32> {
    %0 = stablehlo.broadcast_in_dim %operand, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }
}
