module @jit_eltwise_add attributes {} {
  func.func private @add_impl(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_composite(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %results = stablehlo.composite "jit_eltwise_add.my_add" %arg0, %arg1 {decomposition = @add_impl} : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %results : tensor<64x128xf32>
  }
}
