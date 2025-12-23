module @jit_concatenate attributes {} {
  func.func public @test_concatenate(%arg0: tensor<32x128xf32>, %arg1: tensor<32x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
