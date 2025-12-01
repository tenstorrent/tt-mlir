module {
  func.func @log1p(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %1 = "ttir.log1p"(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
