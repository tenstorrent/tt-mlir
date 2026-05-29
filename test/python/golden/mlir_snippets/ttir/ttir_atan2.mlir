module {
  func.func @atan2(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "ttir.atan2"(%arg0, %arg1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
