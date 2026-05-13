module {
  func.func @square(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "ttir.square"(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
