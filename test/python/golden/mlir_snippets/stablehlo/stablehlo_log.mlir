module @jit_eltwise_log attributes {} {
  func.func public @test_log(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.log %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
