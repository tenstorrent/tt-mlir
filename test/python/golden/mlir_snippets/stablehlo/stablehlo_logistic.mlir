module @jit_eltwise_logistic attributes {} {
  func.func public @test_logistic(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.logistic %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
