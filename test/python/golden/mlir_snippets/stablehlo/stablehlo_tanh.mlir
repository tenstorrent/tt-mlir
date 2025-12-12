module @jit_eltwise_tanh attributes {} {
  func.func public @test_tanh(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
