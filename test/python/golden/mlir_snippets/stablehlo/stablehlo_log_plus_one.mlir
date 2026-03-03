module @jit_eltwise_log_plus_one attributes {} {
  func.func public @test_log_plus_one(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.log_plus_one %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
