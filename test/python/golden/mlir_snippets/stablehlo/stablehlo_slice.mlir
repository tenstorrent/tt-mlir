module @jit_tensor_slice attributes {} {
  func.func public @test_slice(%arg0: tensor<128x128xf32>) -> tensor<64x64xf32> {
    %0 = "stablehlo.slice"(%arg0) {
      start_indices = array<i64: 0, 0>,
      limit_indices = array<i64: 64, 64>,
      strides = array<i64: 1, 1>
    } : (tensor<128x128xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
