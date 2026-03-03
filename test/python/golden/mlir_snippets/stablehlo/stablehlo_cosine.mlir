module @jit_eltwise_cosine attributes {} {
  func.func public @test_cosine(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.cosine %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
