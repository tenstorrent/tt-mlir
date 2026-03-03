module @jit_eltwise_tan attributes {} {
  func.func public @test_tan(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.tan %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
