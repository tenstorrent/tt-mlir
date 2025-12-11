module @jit_eltwise_ceil attributes {} {
  func.func public @test_ceil(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.ceil %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
