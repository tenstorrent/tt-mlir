module {
  func.func @argmax(%arg0: tensor<128x10xf32>) -> tensor<128xsi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10xf32>) -> tensor<128xsi32>
    return %0 : tensor<128xsi32>
  }
}
