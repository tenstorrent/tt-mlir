module {
  func.func @max_0(%arg0: tensor<1x96x32x32xf32>) -> tensor<1x96x32xf32> {
    %0 = "ttir.max"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x96x32x32xf32>) -> tensor<1x96x32xf32>
    return %0 : tensor<1x96x32xf32>
  }
}
