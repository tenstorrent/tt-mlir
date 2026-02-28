module {
  func.func @sum_0(%arg0: tensor<1x32x2048xf32>) -> tensor<1x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x32x2048xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  func.func @sum_1(%arg0: tensor<1x32x512xf32>) -> tensor<1x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x32x512xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  func.func @sum_2(%arg0: tensor<1x32x3072xf32>) -> tensor<1x32xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x32x3072xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  func.func @sum_3(%arg0: tensor<1x32x16x32xbf16>) -> tensor<1x32x16xbf16> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x32x16x32xbf16>) -> tensor<1x32x16xbf16>
    return %0 : tensor<1x32x16xbf16>
  }
}
