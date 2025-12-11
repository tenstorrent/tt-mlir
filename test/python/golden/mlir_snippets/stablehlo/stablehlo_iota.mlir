module @jit_iota attributes {} {
  func.func public @test_iota() -> tensor<4x8xf32> {
    %0 = stablehlo.iota dim = 1 : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}
