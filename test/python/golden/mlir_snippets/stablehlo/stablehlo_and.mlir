module @jit_eltwise_and attributes {} {
  func.func public @test_and(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = stablehlo.and %arg0, %arg1 : tensor<64x128xi32>
    return %0 : tensor<64x128xi32>
  }
}
