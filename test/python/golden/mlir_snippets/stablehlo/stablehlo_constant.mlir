module @jit_constant attributes {} {
  func.func public @test_constant() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
