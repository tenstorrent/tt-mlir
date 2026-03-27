module {
  func.func @sum_0(%arg0: tensor<1x32x7168xf32>) -> tensor<1x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x32x7168xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  func.func @sum_1(%arg0: tensor<1x32x1536xf32>) -> tensor<1x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x32x1536xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  func.func @sum_2(%arg0: tensor<1x32x512xf32>) -> tensor<1x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x32x512xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  func.func @sum_3(%arg0: tensor<1x128x32x32xf32>) -> tensor<1x128x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x128x32x32xf32>) -> tensor<1x128x32xf32>
    return %0 : tensor<1x128x32xf32>
  }
}
