module {
  func.func @max_0(%arg0: tensor<1x16x128x129xbf16>) -> tensor<1x16x128xbf16> {
    %0 = "ttir.max"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128xbf16>
    return %0 : tensor<1x16x128xbf16>
  }

  func.func @max_1(%arg0: tensor<128x4xbf16>) -> tensor<128xbf16> {
    %0 = "ttir.max"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x4xbf16>) -> tensor<128xbf16>
    return %0 : tensor<128xbf16>
  }
}
