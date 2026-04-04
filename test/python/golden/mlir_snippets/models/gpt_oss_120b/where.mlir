module {
  func.func @where_0(%arg0: tensor<1x1x128x128xi1>, %arg1: tensor<1x1x128x128xbf16>, %arg2: tensor<1x1x128x128xbf16>) -> tensor<1x1x128x128xbf16> {
    %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<1x1x128x128xi1>, tensor<1x1x128x128xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x1x128x128xbf16>
    return %0 : tensor<1x1x128x128xbf16>
  }
}
