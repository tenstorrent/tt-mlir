module {
  func.func public @reduce_not_keep_dim(%arg0: tensor<128x10xf32>) -> tensor<128xf32> {
    %1 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10xf32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
}
