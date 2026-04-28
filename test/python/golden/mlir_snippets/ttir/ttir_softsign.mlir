module {
  func.func @softsign(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "ttir.softsign"(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
