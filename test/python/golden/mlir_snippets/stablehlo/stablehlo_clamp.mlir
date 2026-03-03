module @jit_eltwise_clamp attributes {} {
  func.func public @test_clamp(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
