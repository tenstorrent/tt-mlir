module @jit_dot_general attributes {} {
  func.func public @test_dot_general(%arg0: tensor<64x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<64x256xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
}
