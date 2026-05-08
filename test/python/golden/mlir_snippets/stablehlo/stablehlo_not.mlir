module @jit_not attributes {} {
  func.func public @test_logical_not(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %0 = stablehlo.not %arg0 : tensor<4xi1>
    return %0 : tensor<4xi1>
  }
  func.func public @test_bitwise_not(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %0 = stablehlo.not %arg0 : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
  func.func public @test_not_2d(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %0 = stablehlo.not %arg0 : tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
  func.func public @test_not_bool_2d(%arg0: tensor<3x4xi1>) -> tensor<3x4xi1> {
    %0 = stablehlo.not %arg0 : tensor<3x4xi1>
    return %0 : tensor<3x4xi1>
  }
}
