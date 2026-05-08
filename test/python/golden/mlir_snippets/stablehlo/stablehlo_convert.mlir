module @jit_eltwise_convert attributes {} {
  func.func public @test_convert(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<64x128xf32>) -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}
