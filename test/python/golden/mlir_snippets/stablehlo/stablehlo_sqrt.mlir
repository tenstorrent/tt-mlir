module @jit_eltwise_sqrt attributes {} {
  func.func public @test_sqrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.sqrt %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
