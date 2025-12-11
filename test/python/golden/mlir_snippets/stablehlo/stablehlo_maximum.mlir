module @jit_eltwise_maximum attributes {} {
  func.func public @test_maximum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
