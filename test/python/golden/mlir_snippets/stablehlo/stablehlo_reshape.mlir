module @jit_tensor_reshape attributes {} {
  func.func public @test_reshape(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}
