module @jit_tensor_reverse attributes {} {
  func.func public @test_reverse(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "stablehlo.reverse" (%arg0) {dimensions = array<i64: 1>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
