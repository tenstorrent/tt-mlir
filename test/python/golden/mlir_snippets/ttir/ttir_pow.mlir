module {
  func.func @pow(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "ttir.pow"(%arg0, %arg1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
