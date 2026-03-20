module {
  func.func @asin(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "ttir.asin"(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
