module @jit_tensor_select attributes {} {
  func.func public @test_select(%pred: tensor<64x128xi1>, %t: tensor<64x128xf32>, %f: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.select %pred, %t, %f : (tensor<64x128xi1>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
