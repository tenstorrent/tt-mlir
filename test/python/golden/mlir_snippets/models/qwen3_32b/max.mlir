module {
  func.func @max_0(%arg0: tensor<32x8x17x128xf32>) -> tensor<32x8x17xf32> {
    %0 = "ttir.max"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x8x17x128xf32>) -> tensor<32x8x17xf32>
    return %0 : tensor<32x8x17xf32>
  }
}
