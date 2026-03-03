module @jit_eltwise_rsqrt attributes {} {
  func.func public @test_rsqrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.rsqrt %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
