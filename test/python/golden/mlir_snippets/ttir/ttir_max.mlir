module {
  func.func @forward(%arg0: tensor<32x32x64xbf16>) -> tensor<1x32x64xbf16> {
    %1 = "ttir.max"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<32x32x64xbf16>) -> tensor<1x32x64xbf16>
    return %1 : tensor<1x32x64xbf16>
  }
}
