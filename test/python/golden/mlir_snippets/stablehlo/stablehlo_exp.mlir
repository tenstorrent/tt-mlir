module @jit_eltwise_exp attributes {} {
  func.func public @test_exp(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.exponential %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
