module @jit_eltwise_sine attributes {} {
  func.func public @test_sine(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.sine %arg0 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
