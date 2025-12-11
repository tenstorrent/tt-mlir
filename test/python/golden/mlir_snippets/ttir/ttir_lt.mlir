module {
  func.func @comparison_ops(%arg0: tensor<128x128xbf16>, %arg1: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %0 = "ttir.lt"(%arg0, %arg1) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %0 : tensor<128x128xbf16>
  }
}
