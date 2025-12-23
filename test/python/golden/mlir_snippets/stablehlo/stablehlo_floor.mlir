module @jit_eltwise_floor attributes {} {
  func.func public @test_floor(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.floor %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
