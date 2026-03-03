module @jit_eltwise_shift_right_logical attributes {} {
  func.func public @test_shift_right_logical(%arg0: tensor<64x128xui32>, %arg1: tensor<64x128xui32>) -> tensor<64x128xui32> {
    %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<64x128xui32>
    return %0 : tensor<64x128xui32>
  }
}
