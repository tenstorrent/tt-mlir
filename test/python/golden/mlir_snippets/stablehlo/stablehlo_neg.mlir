module @jit_eltwise_neg attributes {} {
  func.func public @test_neg(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.negate %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
