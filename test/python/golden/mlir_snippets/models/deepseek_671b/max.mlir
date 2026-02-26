module {
  func.func @max_0(%arg0: tensor<1x32x16x32xbf16>) -> tensor<1x32x16xbf16> {
    %0 = "ttir.max"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x32x16x32xbf16>) -> tensor<1x32x16xbf16>
    return %0 : tensor<1x32x16xbf16>
  }
}
