module @jit_dot_general attributes {} {
  func.func public @test_dot_general_2d(%arg0: tensor<16x32xf32>, %arg1: tensor<32x8xf32>) -> tensor<16x8xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}
