module {
  func.func @sum_0(%arg0: tensor<1x32x5120xf32>) -> tensor<1x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x32x5120xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  func.func @sum_1(%arg0: tensor<1x32x8x128xf32>) -> tensor<1x32x8xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x32x8x128xf32>) -> tensor<1x32x8xf32>
    return %0 : tensor<1x32x8xf32>
  }

  func.func @sum_2(%arg0: tensor<1x32x96x128xf32>) -> tensor<1x32x96xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x32x96x128xf32>) -> tensor<1x32x96xf32>
    return %0 : tensor<1x32x96xf32>
  }

  func.func @sum_3(%arg0: tensor<1x96x32x32xf32>) -> tensor<1x96x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x96x32x32xf32>) -> tensor<1x96x32xf32>
    return %0 : tensor<1x96x32xf32>
  }
}
