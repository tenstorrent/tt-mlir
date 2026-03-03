module @jit_eltwise_abs attributes {} {
  func.func public @test_abs(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.abs %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
