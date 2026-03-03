module {
  func.func @typecast(%arg0: tensor<128x128xf32>) -> tensor<128x128xf64> {
    %1 = "ttir.typecast"(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf64>
    return %1 : tensor<128x128xf64>
  }
}
