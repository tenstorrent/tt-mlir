module @jit_compare attributes {} {
  func.func public @test_compare(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xi1> {
    %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }
}
