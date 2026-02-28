module {
  func.func @sum_0(%arg0: tensor<1x128x360xf32>) -> tensor<1x128xf32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x128x360xf32>) -> tensor<1x128xf32>
    return %0 : tensor<1x128xf32>
  }

  func.func @sum_1(%arg0: tensor<1x16x128x129xbf16>) -> tensor<1x16x128xbf16> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128xbf16>
    return %0 : tensor<1x16x128xbf16>
  }

  func.func @sum_2(%arg0: tensor<128x4xbf16>) -> tensor<128xbf16> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x4xbf16>) -> tensor<128xbf16>
    return %0 : tensor<128xbf16>
  }

  func.func @sum_3(%arg0: tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16>
    return %0 : tensor<1x128x360xbf16>
  }
}
